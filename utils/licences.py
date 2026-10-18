from datetime import date

import pandas as pd
import pyspark.sql.functions as F

from utils.bronze import run_id_for
from utils.gold import grid_df, read_silver
from utils.silver import read_bronze_batch, write_silver


def process_silver_licences(start, end, config, spark):
    """The licences of the venue groups in config, with the dates that make a site active."""
    run_id = run_id_for(start, end)
    bronze = read_bronze_batch("licences", start, end, config, spark)

    # each licence type belongs to at most one group (config features.licence_groups); others are left out
    group = F.lit(None).cast("string")
    for name, types in config["features"]["licence_groups"].items():
        group = F.when(F.col("license_description").isin(types), F.lit(name)).otherwise(group)

    df = bronze.select(
        # a site is one venue: an account can hold several sites, and a site renews its licence every year or two
        "account_number", "site_number",
        F.col("community_area").cast("int").alias("community_area"),
        group.alias("licence_group"),
        F.col("license_start_date").cast("timestamp_ntz").alias("license_start_date"),
        F.col("expiration_date").cast("timestamp_ntz").alias("expiration_date"),
        "_arrived_at", "date", "run_id",
    ).filter(F.col("licence_group").isNotNull() & F.col("community_area").isNotNull())

    # save silver table to datamart - IRL write to the data lake (S3)
    silver_path, written = write_silver(df, "licences", run_id, config, spark)
    print(f"silver licences {run_id}: {bronze.count():,} licences in bronze, "
          f"{written:,} in a venue group with an area written to {silver_path}")
    return written


def lic_features(targets, config, spark):
    """lic_: per area, the sites of each venue group with a licence active at T, as known at T:
    issued before T (date_issued = _arrived_at), started on or before T, and not expired."""
    last = targets.max().to_pydatetime()
    # a licence can have been issued years before the window, so read every arrival up to T
    licences = read_silver("licences", date(2000, 1, 1), last.date(), config, spark)
    hours = spark.createDataFrame(pd.DataFrame({"T": targets})).select(F.col("T").cast("timestamp_ntz"))

    active = hours.join(licences, (F.col("_arrived_at") < F.col("T"))
                        & (F.col("license_start_date") <= F.col("T"))
                        & (F.col("expiration_date").isNull() | (F.col("expiration_date") > F.col("T"))))
    groups = list(config["features"]["licence_groups"])
    counts = (active.groupBy("community_area", "T").pivot("licence_group", groups)
              .agg(F.countDistinct("account_number", "site_number")))

    # an area with no active venue is a known 0, not missing
    df = grid_df(targets, spark).withColumnRenamed("hour_start", "T").join(counts, ["community_area", "T"], "left")
    return df.select("community_area", F.col("T").alias("hour_start"),
                     *[F.coalesce(F.col(g), F.lit(0)).cast("double").alias(f"lic_{g}_active") for g in groups])