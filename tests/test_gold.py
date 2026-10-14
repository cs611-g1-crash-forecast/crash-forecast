"""Point-in-time tests for the gold feature store: a feature may use a record
only if it arrived before the target hour T. Run with: python -m pytest tests/"""
from datetime import datetime, timedelta

import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from utils.gold import hour_grid, process_gold_features

T = datetime(2018, 6, 15, 10)  # the target hour every test builds features for
HOUR = timedelta(hours=1)


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
            "window": {"start": "2018-03-01T00:00"},
            "deployment": {"go_live": "2023-01-01T00:00"},
            "features": {"families": ["cal", "crash", "wx"],
                         "covid_regime": {"start": "2020-03", "end": "2021-06"}}}


def write_silver(spark, config, crashes, weather):
    """A tiny silver layer: crashes as (area, occurred, notified), weather as (observed, temp)."""
    c = pd.DataFrame(crashes, columns=["community_area", "crash_date", "_arrived_at"])
    c["crash_type"] = "NO INJURY / DRIVE AWAY"
    w = pd.DataFrame(weather, columns=["observed_at", "temp_c"])
    for col in ["dew_point_c", "visibility_m", "wind_ms", "precip_1h_mm"]:
        w[col] = 0.0
    for name, pdf, arrived in [("crashes", c, "_arrived_at"), ("weather", w, "observed_at")]:
        df = spark.createDataFrame(pdf)
        for col, dtype in df.dtypes:
            if dtype == "timestamp":
                df = df.withColumn(col, F.col(col).cast("timestamp_ntz"))
        (df.withColumn("date", F.to_date(arrived)).withColumn("run_id", F.lit("test"))
           .write.mode("overwrite").partitionBy("date", "run_id").parquet(f"{config['paths']['datamart']}/silver/{name}"))


def features_for(target, spark, config):
    """The feature row of area 1 for one target hour, built by the hourly run [target - 1 h, target)."""
    process_gold_features(target - HOUR, target, config, spark)
    df = spark.read.parquet(f"{config['paths']['datamart']}/gold/feature_store")
    return df.filter((F.col("community_area") == 1) & (F.col("hour_start") == target)).first()


def test_a_crash_notified_at_T_is_not_a_feature_of_T(spark, config):
    write_silver(spark, config,
                 crashes=[(1, T - 2 * HOUR, T - timedelta(minutes=1)),  # notified just before T: counts
                          (1, T - 2 * HOUR, T)],                         # notified exactly at T: not yet known
                 weather=[(T - timedelta(minutes=9), 20.0)])
    row = features_for(T, spark, config)
    assert row.crash_area_notified_1h == 1
    assert row.crash_city_notified_24h == 1
    # one hour later, both are known (within the last 3 h; the 1 h window has moved past the first)
    assert features_for(T + HOUR, spark, config).crash_area_notified_3h == 2


def test_the_weekly_mean_uses_only_crashes_reported_before_T(spark, config):
    write_silver(spark, config,
                 crashes=[(1, T - timedelta(days=7), T - timedelta(days=7)),   # reported at once: counts
                          (1, T - timedelta(days=14), T + timedelta(minutes=5))],  # reported after T: doesn't
                 weather=[(T - timedelta(minutes=9), 20.0)])
    assert features_for(T, spark, config).crash_area_how_mean_4w == 0.25


def test_weather_is_the_last_reading_before_T(spark, config):
    write_silver(spark, config, crashes=[(1, T - HOUR, T - HOUR)],
                 weather=[(T - timedelta(minutes=69), 18.0),
                          (T - timedelta(minutes=9), 20.0),   # the last one before T
                          (T, 25.0)])                         # observed at T: too late
    row = features_for(T, spark, config)
    assert row.wx_temp_c == 20.0
    assert row.feature_cutoff == row.hour_start == T


def test_the_grid_follows_chicago_clock_hours():
    # spring forward: 02:00 doesn't exist; fall back: 01:00 happens twice but is one row
    assert len(hour_grid(datetime(2018, 3, 11, 0), datetime(2018, 3, 11, 23))) == 23
    assert len(hour_grid(datetime(2018, 11, 4, 0), datetime(2018, 11, 4, 23))) == 24
    # a range may start or end on those hours (the bulk run for November 2020 starts at 01:00 on 1 Nov)
    assert len(hour_grid(datetime(2020, 11, 1, 1), datetime(2020, 11, 1, 3))) == 3
    assert len(hour_grid(datetime(2018, 3, 11, 2), datetime(2018, 3, 11, 4))) == 2
    assert len(hour_grid(datetime(2018, 3, 11, 0), datetime(2018, 3, 11, 2))) == 2
