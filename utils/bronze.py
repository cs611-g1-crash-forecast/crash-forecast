import csv
import gzip
import os
from collections import Counter
from datetime import datetime

import glob
from datetime import timedelta

import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta


def source_files(file_pattern, raw_dir, start, end):
    """The raw file(s) that can hold records arriving in [start, end)."""
    if "{month}" not in file_pattern:
        return [os.path.join(raw_dir, file_pattern)]

    # monthly files are split by when a record starts (a taxi trip's start, a tracker reading),
    # and a record can arrive in the next month (a trip from 23:45 on the 31st ends after midnight),
    # so read from the month before start up to the month of end
    files = []
    month = datetime(start.year, start.month, 1) - relativedelta(months=1)
    while month < end:
        path = os.path.join(raw_dir, file_pattern.format(month=month.strftime("%Y-%m")))
        if os.path.exists(path):
            files.append(path)
        month += relativedelta(months=1)
    return files


def landing_files(source, landing_dir, start, end):
    """Live runs read the simulated feed: the landing files of the arrival dates in [start, end)."""
    files = []
    day = start.date()
    while day <= (end - timedelta(microseconds=1)).date():
        files += sorted(glob.glob(os.path.join(landing_dir, source, f"date={day}", "*.csv.gz")))
        day += timedelta(days=1)
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


def read_raw(files, source_cfg, spark):
    """The source rows as text, with the arrival time of each record in _arrived_at."""
    # the header as written (Spark would rename repeated names to NAME1, NAME215, ...)
    with gzip.open(files[0], "rt") as f:
        header = next(csv.reader(f))

    # load data - IRL this is the source system's feed
    # no inferSchema: every column stays text, exactly as the source wrote it
    # multiLine: a quoted field may contain line breaks (the licences' "location" does);
    # escape='"': a quote inside a quoted field is written as two quotes
    df = (spark.read.csv(files, header=True, multiLine=True, escape='"')
          .toDF(*unique_columns(header)))
    for old, new in source_cfg.get("rename", {}).items():
        df = df.withColumnRenamed(old, new)

    # when each record arrives: weather is stamped in UTC, everything else in Chicago time
    arrived = F.col(source_cfg["arrived_at"])
    if source_cfg.get("arrived_at_is_utc"):
        arrived = F.from_utc_timestamp(F.to_timestamp(arrived), "America/Chicago")
    return df.withColumn("_arrived_at", arrived.cast("timestamp_ntz"))


def process_bronze_table(source, start, end, config, spark):
    source_cfg = config["sources"][source]
    raw_dir = config["paths"]["raw_data"]
    window_start = datetime.fromisoformat(config["window"]["start"])

    # bulk runs (before go-live) read the source files; live runs read the simulated feed
    if start < datetime.fromisoformat(config["deployment"]["go_live"]):
        files = source_files(source_cfg["file"], raw_dir, start, end)
    else:
        files = landing_files(source, config["paths"]["landing"], start, end)
    if not files:
        # a live interval with nothing in the feed for this source (no licence is issued on a Sunday)
        print(f"bronze {source} {run_id_for(start, end)}: nothing arrived, 0 rows written")
        return 0

    df = read_raw(files, source_cfg, spark)

    # keep what arrived in [start, end); the first run of the study also takes
    # everything that arrived before it (e.g. licences issued before 2018)
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


def stage_feed(source, start, end, config, spark):
    """SIMULATION ONLY, not part of the production pipeline: split the records that arrive in
    [start, end) into one landing folder per arrival date, as a live feed would deliver them
    (datamart/landing/<source>/date=YYYY-MM-DD/). Live bronze runs read from there."""
    source_cfg = config["sources"][source]
    files = source_files(source_cfg["file"], config["paths"]["raw_data"], start, end)
    df = read_raw(files, source_cfg, spark).filter((F.col("_arrived_at") >= start) & (F.col("_arrived_at") < end))

    # the rows as received (after bronze's renames), one folder per arrival date
    landing_path = os.path.join(config["paths"]["landing"], source)
    (df.withColumn("date", F.to_date("_arrived_at")).drop("_arrived_at")
       .write.mode("overwrite").partitionBy("date")
       .option("header", True).option("escape", '"').option("compression", "gzip")
       .csv(landing_path))
    landed = spark.read.csv(landing_path, header=True, multiLine=True, escape='"')
    print(f"stage_feed {source}: {landed.count():,} rows that arrive from {start:%Y-%m-%d} to {end:%Y-%m-%d} "
          f"written to {landing_path}, {landed.select('date').distinct().count()} arrival dates")
