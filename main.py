import argparse
import time
from datetime import datetime

import yaml
from pyspark.sql import SparkSession

import utils.bronze
import utils.gold
import utils.model
import utils.monitor
import utils.serve
import utils.silver
import utils.tables

STAGES = ["stage_feed", "bronze", "silver", "gold_labels", "gold_features", "train",
          "check_data_quality", "infer", "log_run",
          "monitor_performance", "monitor_stability", "monitor_attribution", "monitor_fairness", "decide_actions"]
SPARK_STAGES = ["stage_feed", "bronze", "silver", "gold_labels", "gold_features"]


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
    if args.stage == "stage_feed":
        return utils.bronze.stage_feed(args.source, start, end, config, spark)
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
    # the stages below use pandas and scikit-learn: their data fits in memory, so no Spark session
    if args.stage == "train":
        return len(utils.model.train_models(start, end, config))
    if args.stage == "check_data_quality":
        return utils.serve.check_data_quality(start, end, config)
    if args.stage == "infer":
        return utils.serve.infer(start, end, args.role, config)
    if args.stage == "log_run":
        return utils.serve.log_run(start, end, config)
    if args.stage == "monitor_performance":
        return utils.monitor.monitor_performance(start, end, config)
    if args.stage == "monitor_stability":
        return utils.monitor.monitor_stability(start, end, config)
    if args.stage == "monitor_attribution":
        return utils.monitor.monitor_attribution(start, end, config)
    if args.stage == "monitor_fairness":
        return utils.monitor.monitor_fairness(start, end, config)
    if args.stage == "decide_actions":
        return utils.monitor.decide_actions(start, end, config)
    raise SystemExit(f"{args.stage}: unknown source {args.source!r}")


def main():
    parser = argparse.ArgumentParser(description="Run one pipeline stage for one interval [start, end).")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--start", required=True, help="interval start, e.g. 2018-03-01T00:00")
    parser.add_argument("--end", required=True, help="interval end (exclusive)")
    parser.add_argument("--source", help="stage_feed, bronze, silver: the source, e.g. crashes")
    parser.add_argument("--role", choices=["production", "shadow"], help="infer: which model in the registry")
    args = parser.parse_args()

    with open("config/pipeline.yaml") as f:
        config = yaml.safe_load(f)
    start, end = parse_time(args.start), parse_time(args.end)

    # time the stage and record it for the run log, also when it fails
    task = "_".join(x for x in [args.stage, args.source, args.role] if x)
    record = {"run_id": utils.bronze.run_id_for(start, end), "task": task,
              "started_at": utils.model.utc_now(), "status": "failed", "rows_out": None}
    began = time.time()
    spark = get_spark(config) if args.stage in SPARK_STAGES and args.source != "boundaries" else None
    try:
        record["rows_out"] = run_stage(args, start, end, config, spark)
        record["status"] = "success"
    finally:
        if spark:
            spark.stop()
        record["seconds"] = round(time.time() - began, 1)
        utils.tables.record_task_run(record, config)


if __name__ == "__main__":
    main()
