"""Point-in-time tests for the taxi_ and lic_ features: a trip counts once it has ended before T,
a licence once it was issued before T and only while it is active."""
from datetime import datetime, timedelta

import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from utils.gold import hour_grid
from utils.licences import lic_features
from utils.taxi import taxi_features

T = datetime(2018, 6, 15, 10)  # the target hour every test builds features for
HOUR, DAY = timedelta(hours=1), timedelta(days=1)


@pytest.fixture(scope="module")
def spark():
    spark = (SparkSession.builder.master("local[1]")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    yield spark
    spark.stop()


@pytest.fixture
def config(tmp_path):
    return {"paths": {"datamart": str(tmp_path)},
            "features": {"licence_groups": {"nightlife": ["Tavern"], "package_goods": ["Package Goods"]}}}


def write_silver(spark, config, name, pdf, date_column):
    df = spark.createDataFrame(pdf)
    for col, dtype in df.dtypes:
        if dtype == "timestamp":
            df = df.withColumn(col, F.col(col).cast("timestamp_ntz"))
    (df.withColumn("date", F.to_date(date_column)).withColumn("run_id", F.lit("test"))
       .write.mode("overwrite").partitionBy("date", "run_id").parquet(f"{config['paths']['datamart']}/silver/{name}"))


def row_for(features, target, spark, config):
    rows = features(hour_grid(target, target), config, spark).collect()
    return {r.community_area: r for r in rows}


def test_taxi_counts_trips_that_ended_in_the_hour_before_T(spark, config):
    trips = pd.DataFrame([(1, T - HOUR, 3), (None, T - HOUR, 2),  # ended in [T-1h, T): known at T
                          (1, T - 2 * HOUR, 7),                   # ended earlier: not in the last hour
                          (1, T, 5)],                             # ended at T or later: not yet known
                         columns=["community_area", "hour_start", "trips"])
    trips["community_area"] = trips["community_area"].astype("Int32")
    trips["_arrived_at"] = trips["hour_start"] + timedelta(minutes=45)
    write_silver(spark, config, "taxi", trips, "hour_start")
    rows = row_for(taxi_features, T, spark, config)
    assert rows[1].taxi_area_trips_1h == 3
    assert rows[1].taxi_city_trips_1h == 5      # the trips without a pickup area count citywide
    assert rows[2].taxi_area_trips_1h == 0      # no trip ended there: a known 0


def licence(site, group, start, expires, issued):
    return {"account_number": "1", "site_number": str(site), "community_area": 1, "licence_group": group,
            "license_start_date": start, "expiration_date": expires, "_arrived_at": issued}


def test_a_licence_counts_only_once_issued_and_while_active(spark, config):
    year = 365 * DAY
    pdf = pd.DataFrame([
        licence(1, "nightlife", T - year, T + year, T - year),           # active, issued long ago: counts
        licence(1, "nightlife", T + DAY, T + 2 * year, T - 30 * DAY),    # the same site's renewal: still one site
        licence(2, "nightlife", T - DAY, T + year, T + 5 * DAY),         # started, but issued after T: not known yet
        licence(3, "nightlife", T - 2 * year, T - DAY, T - 2 * year),    # expired before T
        licence(4, "package_goods", T - year, T, T - year),              # expires exactly at T: no longer active
        licence(5, "package_goods", T - year, None, T - year),           # no expiry date: active
    ])
    write_silver(spark, config, "licences", pdf, "_arrived_at")
    rows = row_for(lic_features, T, spark, config)
    assert rows[1].lic_nightlife_active == 1
    assert rows[1].lic_package_goods_active == 1
    assert rows[2].lic_nightlife_active == 0