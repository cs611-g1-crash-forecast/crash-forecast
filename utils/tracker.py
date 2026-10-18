import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from shapely import STRtree, points
from shapely.geometry import shape

from utils.bronze import run_id_for
from utils.gold import grid_df, hour_of, read_silver
from utils.silver import read_bronze_batch, write_silver

# many arterials run along an area's edge or the city limit, so their midpoint can fall a few metres
# outside every polygon: such a segment takes the nearest area within 0.001 degrees (about 100 m)
EDGE_TOLERANCE_DEG = 0.001


def segment_areas(segments, boundaries_file):
    """The community area (1-77) of each segment midpoint, or None if it is further than the tolerance from every area."""
    with open(boundaries_file) as f:
        features = json.load(f)["features"]
    polygons = [shape(feature["geometry"]) for feature in features]
    numbers = np.array([int(feature["properties"]["area_numbe"]) for feature in features])
    midpoints = points(segments["longitude"].to_numpy(), segments["latitude"].to_numpy())
    # nearest polygon within the tolerance; a point inside a polygon is at distance 0 from it
    point_index, polygon_index = STRtree(polygons).query_nearest(
        midpoints, max_distance=EDGE_TOLERANCE_DEG, all_matches=False)
    area = np.full(len(segments), None, dtype=object)
    area[point_index] = numbers[polygon_index]
    return area


def process_silver_tracker(start, end, config, spark):
    """Segment x hour: the mean estimated speed and how many readings had one, with the segment's community area."""
    run_id = run_id_for(start, end)
    bronze = read_bronze_batch("tracker", start, end, config, spark)

    readings = bronze.select(
        F.col("segment_id").cast("int").alias("segment_id"),
        F.col("_arrived_at"),
        # -1 means no bus passed, so there is no estimate: missing, never 0 mph (which would read as gridlock)
        F.when(F.col("speed").cast("int") >= 0, F.col("speed").cast("int")).alias("speed"),
        ((F.col("start_latitude").cast("double") + F.col("end_latitude").cast("double")) / 2).alias("latitude"),
        ((F.col("start_longitude").cast("double") + F.col("end_longitude").cast("double")) / 2).alias("longitude"),
    )

    # a reading every ~10 minutes is finer than our hourly grain: aggregate each segment to the hour
    hourly = readings.groupBy("segment_id", hour_of("_arrived_at").alias("hour_start")).agg(
        F.avg("speed").alias("speed_mean"),
        F.count("*").cast("int").alias("readings"),
        F.count("speed").cast("int").alias("readings_with_speed"),
        F.max("_arrived_at").alias("_arrived_at"),
    )

    # each segment's midpoint -> community area, once per segment (about 1,000) rather than once per reading
    segments = (readings.filter(F.col("latitude").isNotNull() & F.col("longitude").isNotNull())
                .groupBy("segment_id").agg(F.first("latitude").alias("latitude"),
                                           F.first("longitude").alias("longitude"))
                .toPandas())
    boundaries_file = os.path.join(config["paths"]["datamart"], "bronze", "boundaries",
                                   config["sources"]["boundaries"]["file"])
    segments["community_area"] = segment_areas(segments, boundaries_file)
    mapped = segments.dropna(subset=["community_area"]).astype({"segment_id": int, "community_area": int})
    areas = spark.createDataFrame(mapped[["segment_id", "community_area"]]).select(
        F.col("segment_id").cast("int"), F.col("community_area").cast("int"))

    df = (hourly.join(areas, "segment_id")
          .withColumn("date", F.to_date("hour_start"))
          .withColumn("run_id", F.lit(run_id)))

    # save silver table to datamart - IRL write to the data lake (S3)
    silver_path, written = write_silver(df, "tracker", run_id, config, spark)
    n_segments, n_mapped = len(segments), len(mapped)
    print(f"silver tracker {run_id}: {bronze.count():,} readings in bronze, {n_segments:,} segments "
          f"({n_segments - n_mapped} more than 100 m from every area), {written:,} segment-hours written to {silver_path}")
    return written


def trk_features(targets, config, spark):
    """trk_: congestion in the hour before T, per area.
    trk_congestion_index: the mean over the area's segments of speed / the segment's usual speed
    (its mean for the same hour of the week over the previous 4 weeks); below 1 = slower than usual.
    trk_coverage: the share of the area's segments (those seen in the 28 days before T's date) with a speed in that hour."""
    first, last = targets.min().to_pydatetime(), targets.max().to_pydatetime()
    silver = read_silver("tracker", (first - timedelta(days=29)).date(), last.date(), config, spark)

    # the hour before T is complete at T: its segment-hour row arrived before T
    last_hour = (silver.filter(F.col("speed_mean").isNotNull())
                 .withColumn("T", F.timestamp_add("HOUR", F.lit(1), F.col("hour_start")))
                 .filter(F.col("T").between(first, last)))

    # a segment's speed in hour h counts towards its usual speed at h + 7, 14, 21, 28 days
    usual = (silver.filter(F.col("speed_mean").isNotNull())
             .withColumn("k", F.explode(F.sequence(F.lit(1), F.lit(4))))
             .withColumn("hour_start", F.timestamp_add("DAY", F.col("k") * 7, F.col("hour_start")))
             .groupBy("segment_id", "hour_start").agg(F.avg("speed_mean").alias("usual_speed"))
             # an hour can average 0 mph (rare: 0.015% of segment-hours); if every week did, there is no usual speed to divide by
             .filter(F.col("usual_speed") > 0))

    ratios = last_hour.join(usual, ["segment_id", "hour_start"]).withColumn(
        "ratio", F.col("speed_mean") / F.col("usual_speed"))
    index = ratios.groupBy("community_area", "T").agg(F.avg("ratio").alias("trk_congestion_index"))
    reporting = last_hour.groupBy("community_area", "T").agg(F.countDistinct("segment_id").alias("reporting"))

    # the area's segments: those with any reading on the 28 days before T's date
    days = spark.createDataFrame(pd.DataFrame({"day": sorted(set(targets.normalize()))})).select(
        F.to_date("day").alias("day"))
    seen = silver.select("community_area", "segment_id", F.to_date("hour_start").alias("seen")).distinct()
    known = (days.join(seen, (F.col("seen") < F.col("day")) & (F.col("seen") >= F.date_sub("day", 28)))
             .groupBy("community_area", "day").agg(F.countDistinct("segment_id").alias("known")))

    grid = grid_df(targets, spark).withColumnRenamed("hour_start", "T").withColumn("day", F.to_date("T"))
    df = (grid.join(index, ["community_area", "T"], "left")
          .join(reporting, ["community_area", "T"], "left")
          .join(known, ["community_area", "day"], "left"))
    # an area with no segments has no trk_ values: missing, not 0
    return df.select("community_area", F.col("T").alias("hour_start"),
                     F.col("trk_congestion_index").cast("double"),
                     # a segment first seen on T's own date could push the share past 1: cap it
                     # (least() skips nulls, so test for a known segment first)
                     F.when(F.col("known").isNotNull(),
                            F.least(F.coalesce(F.col("reporting"), F.lit(0)) / F.col("known"), F.lit(1.0)))
                     .cast("double").alias("trk_coverage"))