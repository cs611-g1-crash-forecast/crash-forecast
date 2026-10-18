"""Point-in-time tests for the trk_ features: they may use only segment-hours that ended before T."""
from datetime import datetime, timedelta

import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from utils.gold import hour_grid
from utils.tracker import trk_features

T = datetime(2018, 6, 15, 10)  # the target hour every test builds features for
HOUR, WEEK = timedelta(hours=1), timedelta(days=7)


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
    return {"paths": {"datamart": str(tmp_path)}}


def write_silver(spark, config, rows):
    """A tiny silver tracker table: (segment_id, community_area, hour_start, speed_mean or None)."""
    pdf = pd.DataFrame(rows, columns=["segment_id", "community_area", "hour_start", "speed_mean"])
    pdf["readings"] = 6
    pdf["readings_with_speed"] = pdf["speed_mean"].notna() * 6
    pdf["_arrived_at"] = pdf["hour_start"] + timedelta(minutes=55)
    df = spark.createDataFrame(pdf)
    (df.withColumn("hour_start", F.col("hour_start").cast("timestamp_ntz"))
       .withColumn("_arrived_at", F.col("_arrived_at").cast("timestamp_ntz"))
       .withColumn("speed_mean", F.col("speed_mean").cast("double"))
       .withColumn("date", F.to_date("hour_start")).withColumn("run_id", F.lit("test"))
       .write.mode("overwrite").partitionBy("date", "run_id").parquet(f"{config['paths']['datamart']}/silver/tracker"))


def features_for(target, spark, config):
    rows = trk_features(hour_grid(target, target), config, spark).collect()
    return {r.community_area: r for r in rows}


def test_only_the_hour_before_T_counts(spark, config):
    write_silver(spark, config, [
        (10, 1, T - HOUR - WEEK, 20.0),       # usual speed for this hour of the week: 20
        (10, 1, T - HOUR - 2 * WEEK, 20.0),
        (10, 1, T - HOUR, 10.0),              # the hour before T: half the usual speed
        (10, 1, T, 40.0),                     # the hour starting at T: not over yet, must not count
    ])
    assert features_for(T, spark, config)[1].trk_congestion_index == 0.5


def test_coverage_counts_segments_without_a_speed(spark, config):
    write_silver(spark, config, [
        (10, 1, T - HOUR - WEEK, 20.0),
        (10, 1, T - HOUR, 20.0),              # reports a speed in the hour before T
        (11, 1, T - 3 * 24 * HOUR, 25.0),     # seen 3 days ago, but no bus in the hour before T
        (11, 1, T - HOUR, None),
    ])
    row = features_for(T, spark, config)[1]
    assert row.trk_coverage == 0.5
    assert row.trk_congestion_index == 1.0


def test_an_area_without_segments_is_missing_not_zero(spark, config):
    write_silver(spark, config, [(10, 1, T - HOUR - WEEK, 20.0), (10, 1, T - HOUR, 20.0)])
    row = features_for(T, spark, config)[2]
    assert row.trk_congestion_index is None and row.trk_coverage is None


def test_a_segment_usually_at_0_mph_has_no_ratio(spark, config):
    write_silver(spark, config, [
        (10, 1, T - HOUR - WEEK, 20.0), (10, 1, T - HOUR, 10.0),   # half its usual speed
        (12, 1, T - HOUR - WEEK, 0.0), (12, 1, T - HOUR, 15.0),    # usual 0 mph: no ratio, rather than a division by 0
    ])
    row = features_for(T, spark, config)[1]
    assert row.trk_congestion_index == 0.5
    assert row.trk_coverage == 1.0                               # it still reported a speed