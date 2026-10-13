import argparse
from datetime import datetime

import yaml
from pyspark.sql import SparkSession

import utils.bronze
import utils.gold
import utils.silver

STAGES = ["bronze", "silver", "gold_labels", "gold_features"]
SPARK_STAGES = ["bronze", "silver", "gold_labels", "gold_features"]


def get_spark(config):
    spark = (SparkSession.builder
             .appName("crash-forecast")
             .master("local[*]")
             .config("spark.driver.memory", config["spark"]["driver_memory"])
             # timestamps are stored as naive Chicago time, so Spark must not shift them
             .config("spark.sql.session.timeZone", "UTC")
             # overwrite only the partitions a run writes, not the whole table
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.ui.showConsoleProgress", "false")
             # no web UI: parallel jobs would otherwise compete for its port
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def parse_time(text):
    # Airflow passes interval bounds like 2018-03-01T00:00:00+00:00; we use the clock time only
    return datetime.fromisoformat(text).replace(tzinfo=None)


def run_stage(args, start, end, config, spark):
    """Run one stage; returns the number of rows it wrote."""
    if args.stage == "bronze":
        if args.source == "boundaries":
            return utils.bronze.process_bronze_boundaries(config)
        return utils.bronze.process_bronze_table(args.source, start, end, config, spark)
    if args.stage == "silver":
        if args.source == "crashes":
            return utils.silver.process_silver_crashes(start, end, config, spark)
        if args.source == "weather":
            return utils.silver.process_silver_weather(start, end, config, spark)
    if args.stage == "gold_labels":
        return utils.gold.process_gold_labels(start, end, config, spark)
    if args.stage == "gold_features":
        return utils.gold.process_gold_features(start, end, config, spark)
    raise SystemExit(f"{args.stage}: unknown source {args.source!r}")


def main():
    parser = argparse.ArgumentParser(description="Run one pipeline stage for one interval [start, end).")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--start", required=True, help="interval start, e.g. 2018-03-01T00:00")
    parser.add_argument("--end", required=True, help="interval end (exclusive)")
    parser.add_argument("--source", help="bronze, silver: the source, e.g. crashes")
    args = parser.parse_args()

    with open("config/pipeline.yaml") as f:
        config = yaml.safe_load(f)
    start, end = parse_time(args.start), parse_time(args.end)

    spark = get_spark(config) if args.stage in SPARK_STAGES and args.source != "boundaries" else None
    run_stage(args, start, end, config, spark)
    if spark:
        spark.stop()


if __name__ == "__main__":
    main()
