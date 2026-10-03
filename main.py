import argparse
from datetime import datetime

import yaml
from pyspark.sql import SparkSession

import utils.bronze

def get_spark(config):
    spark = (SparkSession.builder
             .appName("crash-forecast")
             .master("local[*]")
             .config("spark.driver.memory", config["spark"]["driver_memory"])
             # fix timestamp to UTC to avoid timezone issues when reading/writing parquet files
             .config("spark.sql.session.timeZone", "UTC")
             # overwrite only the partitions that are written, not the whole table
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def parse_time(text):
    # use clock time to avoid timezone issues when reading/writing parquet files (parquet timestamps are always UTC)
    return datetime.fromisoformat(text).replace(tzinfo=None)

def main():
    parser = argparse.ArgumentParser(description="Run one pipeline stage for one interval [start, end).")
    parser.add_argument("stage", choices=["bronze"])
    parser.add_argument("--start", required=True, help="interval start, e.g. 2018-03-01T00:00")
    parser.add_argument("--end", required=True, help="interval end (exclusive)")
    parser.add_argument("--source", help="bronze only: crashes, weather, tracker, taxi, licences or boundaries")
    args = parser.parse_args()

    with open("config/pipeline.yaml") as f:
        config = yaml.safe_load(f)
    start, end = parse_time(args.start), parse_time(args.end)

    if args.stage == "bronze":
        if args.source == "boundaries":
            utils.bronze.process_bronze_boundaries(config)
        else:
            spark = get_spark(config)
            utils.bronze.process_bronze_table(args.source, start, end, config, spark)
            spark.stop()

if __name__ == "__main__":
    main()


