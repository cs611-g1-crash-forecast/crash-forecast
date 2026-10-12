import json
import os

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StructField, StructType
from shapely import STRtree, points
from shapely.geometry import shape

from utils.bronze import run_id_for


def read_bronze_batch(source, start, end, config, spark):
    """The rows the bronze run for [start, end) wrote: silver processes exactly that batch."""
    bronze_path = os.path.join(config["paths"]["datamart"], "bronze", source)
    return spark.read.parquet(bronze_path).filter(F.col("run_id") == run_id_for(start, end))


def write_silver(df, source, run_id, config, spark):
    # same layout as bronze: arrival date, then run; a rerun replaces only its own partitions
    silver_path = os.path.join(config["paths"]["datamart"], "silver", source)
    df.write.mode("overwrite").partitionBy("date", "run_id").parquet(silver_path)
    written = spark.read.parquet(silver_path).filter(F.col("run_id") == run_id).count()
    return silver_path, written


def community_area_mapper(boundaries_file):
    """A mapInPandas function that adds the community area (1-77) each crash lies in, or null."""
    with open(boundaries_file) as f:
        features = json.load(f)["features"]
    polygons = [shape(feature["geometry"]) for feature in features]
    numbers = np.array([int(feature["properties"]["area_numbe"]) for feature in features])

    def add_community_area(batches):
        tree = STRtree(polygons)
        for pdf in batches:
            crash_points = points(pdf["longitude"].to_numpy(), pdf["latitude"].to_numpy())
            # pairs (point index, polygon index) for every point that lies within a polygon
            point_index, polygon_index = tree.query(crash_points, predicate="within")
            area = np.full(len(pdf), -1)
            area[point_index] = numbers[polygon_index]
            pdf["community_area"] = pd.array(np.where(area > 0, area, None), dtype="Int32")
            yield pdf

    return add_community_area


def process_silver_crashes(start, end, config, spark):
    run_id = run_id_for(start, end)
    bronze = read_bronze_batch("crashes", start, end, config, spark)

    # only the columns the label and the crash features need; post-crash details stay in bronze
    df = bronze.select(
        "crash_record_id",
        F.col("crash_date").cast("timestamp_ntz").alias("crash_date"),
        F.col("date_police_notified").cast("timestamp_ntz").alias("date_police_notified"),
        "crash_type",
        F.col("latitude").cast("double").alias("latitude"),
        F.col("longitude").cast("double").alias("longitude"),
        "_arrived_at", "date", "run_id",
    )

    # 0.7% of crashes have no location (null, or 0 as a placeholder): they can't be placed in an area
    has_location = (F.col("latitude").isNotNull() & (F.col("latitude") != 0)
                    & F.col("longitude").isNotNull() & (F.col("longitude") != 0))
    located = df.filter(has_location)

    # point-in-polygon against the 77 community areas, batch by batch in pandas
    boundaries_file = os.path.join(config["paths"]["datamart"], "bronze", "boundaries",
                                   config["sources"]["boundaries"]["file"])
    schema = StructType(located.schema.fields + [StructField("community_area", IntegerType())])
    mapped = located.mapInPandas(community_area_mapper(boundaries_file), schema)
    in_an_area = mapped.filter(F.col("community_area").isNotNull())

    # save silver table to datamart - IRL write to the data lake (S3)
    silver_path, written = write_silver(in_an_area, "crashes", run_id, config, spark)
    total, with_location = df.count(), located.count()
    print(f"silver crashes {run_id}: {total:,} in bronze, {total - with_location:,} without location, "
          f"{with_location - written:,} outside every area, {written:,} rows written to {silver_path}")
    return written


def isd_value(column, index, missing, scale):
    """One part of a packed ISD field, scaled; null for the missing code (TMP '+0028,5' -> 2.8)."""
    value = F.split(F.col(column), ",").getItem(index).cast("double")
    return F.when(value != missing, value / scale)


def process_silver_weather(start, end, config, spark):
    run_id = run_id_for(start, end)
    bronze = read_bronze_batch("weather", start, end, config, spark)

    # FM-15 is the routine hourly report, taken at :51; the specials and summaries are left out
    hourly = bronze.filter(F.trim("REPORT_TYPE") == "FM-15")

    df = hourly.select(
        F.col("_arrived_at").alias("observed_at"),                                # Chicago time (bronze)
        F.to_timestamp("DATE_UTC").cast("timestamp_ntz").alias("observed_at_utc"),  # as the source wrote it
        isd_value("TMP", 0, 9999, 10).alias("temp_c"),
        isd_value("DEW", 0, 9999, 10).alias("dew_point_c"),
        isd_value("VIS", 0, 999999, 1).alias("visibility_m"),
        isd_value("WND", 3, 9999, 10).alias("wind_ms"),
        isd_value("AA1", 1, 9999, 10).alias("precip_1h_mm"),  # AA1 is absent in 1% of reports: null
        "_arrived_at", "date", "run_id",
    )

    silver_path, written = write_silver(df, "weather", run_id, config, spark)
    print(f"silver weather {run_id}: {bronze.count():,} in bronze, "
          f"{written:,} FM-15 rows written to {silver_path}")
    return written
