from datetime import timedelta

import pyspark.sql.functions as F

from utils.bronze import run_id_for
from utils.gold import grid_df, hour_of, read_silver
from utils.silver import read_bronze_batch, write_silver


def process_silver_taxi(start, end, config, spark):
    """Trips per pickup area and the hour they ended (a trip is known once it ends)."""
    run_id = run_id_for(start, end)
    bronze = read_bronze_batch("taxi", start, end, config, spark)

    # 7% of trips have no pickup area (started outside the city, or suppressed for privacy):
    # they stay as a null area, so they count citywide but in no area
    trips = bronze.select(F.col("pickup_community_area").cast("int").alias("community_area"), "_arrived_at")
    df = (trips.groupBy("community_area", hour_of("_arrived_at").alias("hour_start"))
          .agg(F.count("*").cast("int").alias("trips"), F.max("_arrived_at").alias("_arrived_at"))
          .withColumn("date", F.to_date("hour_start"))
          .withColumn("run_id", F.lit(run_id)))

    # save silver table to datamart 
    silver_path, written = write_silver(df, "taxi", run_id, config, spark)
    total = bronze.count()
    no_area = trips.filter(F.col("community_area").isNull()).count()
    print(f"silver taxi {run_id}: {total:,} trips in bronze, {no_area:,} without a pickup area "
          f"({no_area / max(total, 1):.1%}), {written:,} area-hours written to {silver_path}")
    return written


def taxi_features(targets, config, spark):
    """taxi_: trips that ended in the hour before T, by pickup area and citywide."""
    first, last = targets.min().to_pydatetime(), targets.max().to_pydatetime()
    silver = read_silver("taxi", (first - timedelta(days=1)).date(), last.date(), config, spark)

    # trips that ended in [T - 1 h, T) are known at T
    ended = (silver.withColumn("T", F.timestamp_add("HOUR", F.lit(1), F.col("hour_start")))
             .filter(F.col("T").between(first, last)))
    area = (ended.filter(F.col("community_area").isNotNull())
            .groupBy("community_area", "T").agg(F.sum("trips").cast("double").alias("taxi_area_trips_1h")))
    city = ended.groupBy("T").agg(F.sum("trips").cast("double").alias("taxi_city_trips_1h"))

    # no trip ended is a known 0, not missing
    df = (grid_df(targets, spark).withColumnRenamed("hour_start", "T")
          .join(area, ["community_area", "T"], "left").join(city, ["T"], "left")
          .fillna(0.0, ["taxi_area_trips_1h", "taxi_city_trips_1h"]))
    return df.select("community_area", F.col("T").alias("hour_start"), "taxi_area_trips_1h", "taxi_city_trips_1h")