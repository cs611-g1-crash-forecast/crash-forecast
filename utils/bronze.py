import csv
import gzip
import os
from collections import Counter
from datetime import datetime

import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta


def source_files(file_pattern, raw_dir, start, end):
    """The raw file(s) that can hold records arriving in [start, end)."""
    if "{month}" not in file_pattern:
        return [os.path.join(raw_dir, file_pattern)]

    # Monthly files are split by when a record starts (a taxi trip's start, a tracker reading).
    # A record can arrive in the next month (a trip from 23:45 on the 31st ends after midnight),
    # so read from the month before start up to the month of end
    files = []
    month = datetime(start.year, start.month, 1) - relativedelta(months=1)
    while month < end:
        path = os.path.join(raw_dir, file_pattern.format(month=month.strftime("%Y-%m")))
        if os.path.exists(path):
            files.append(path)
        month += relativedelta(months=1)
    return files


def unique_columns(columns):
    """Number repeated header names (NAME, NAME -> NAME, NAME__2) so Parquet can store them."""
    seen = Counter()
    names = []
    for c in columns:
        seen[c] += 1
        names.append(c if seen[c] == 1 else f"{c}__{seen[c]}")
    return names


def run_id_for(start, end):
    """bulk_YYYYMM for a month-long run, h_YYYYMMDDTHH for an hour, d_YYYYMMDD for a day."""
    hours = (end - start).total_seconds() / 3600
    if hours == 1:
        return "h_" + start.strftime("%Y%m%dT%H")
    if hours == 24:
        return "d_" + start.strftime("%Y%m%d")
    return "bulk_" + start.strftime("%Y%m")


def process_bronze_table(source, start, end, config, spark):
    source_cfg = config["sources"][source]
    raw_dir = config["paths"]["raw_data"]
    window_start = datetime.fromisoformat(config["window"]["start"])

    files = source_files(source_cfg["file"], raw_dir, start, end)

    # the header as written (Spark would rename repeated names to NAME1, NAME215, ...)
    with gzip.open(files[0], "rt") as f:
        header = next(csv.reader(f))

    # load data - IRL this is the source system's feed
    # no inferSchema: every column stays text, exactly as the source wrote it
    # multiLine: a quoted field may contain line breaks (the "location" of the licences do);
    # escape='"': a quote inside a quoted field is written as two quotes
    df = (spark.read.csv(files, header=True, multiLine=True, escape='"')
          .toDF(*unique_columns(header)))
    for old, new in source_cfg.get("rename", {}).items():
        df = df.withColumnRenamed(old, new)

    # when each record arrives: weather is stamped in UTC, everything else in Chicago time
    arrived = F.col(source_cfg["arrived_at"])
    if source_cfg.get("arrived_at_is_utc"):
        arrived = F.from_utc_timestamp(F.to_timestamp(arrived), "America/Chicago")
    df = df.withColumn("_arrived_at", arrived.cast("timestamp_ntz"))

    # keep what arrived in [start, end)
    # the first run of the study also takes everything that arrived before it (eg. licences issued before 2018)
    in_interval = F.col("_arrived_at") < end
    if start > window_start:
        in_interval = in_interval & (F.col("_arrived_at") >= start)
    df = df.filter(in_interval)

    run_id = run_id_for(start, end)
    df = (df.withColumn("_source_file", F.col("_metadata.file_name"))
            .withColumn("_ingested_at", F.current_timestamp().cast("timestamp_ntz"))
            .withColumn("date", F.to_date("_arrived_at"))
            .withColumn("run_id", F.lit(run_id)))

    # save bronze table to datamart - IRL write to the data lake (S3)
    # partitioned by arrival date, then run; a rerun replaces only its own partitions
    bronze_path = os.path.join(config["paths"]["datamart"], "bronze", source)
    df.write.mode("overwrite").partitionBy("date", "run_id").parquet(bronze_path)

    written = spark.read.parquet(bronze_path).filter(F.col("run_id") == run_id).count()
    print(f"bronze {source} {run_id}: {len(files)} file(s) read, {written:,} rows written to {bronze_path}")
    return written


def process_bronze_boundaries(config):
    """Copy the static community-area boundary file into bronze, unchanged."""
    source = os.path.join(config["paths"]["reference"], config["sources"]["boundaries"]["file"])
    bronze_dir = os.path.join(config["paths"]["datamart"], "bronze", "boundaries")
    os.makedirs(bronze_dir, exist_ok=True)
    with open(source, "rb") as f_in, open(os.path.join(bronze_dir, os.path.basename(source)), "wb") as f_out:
        f_out.write(f_in.read())
    print("bronze boundaries: copied", source)


