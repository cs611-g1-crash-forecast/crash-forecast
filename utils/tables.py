import json
import os

import pandas as pd


def table_path(table, config):
    return os.path.join(config["paths"]["datamart"], "gold", table)


def write_rows(df, table, date, run_id, name, config):
    """Write a small table's rows for one (date, run_id) partition as <name>.parquet.
    Several tasks write into the same run's partition (infer_production and infer_shadow), so each
    replaces only its own file; a rerun of a task replaces exactly what it wrote before."""
    folder = os.path.join(table_path(table, config), f"date={date}", f"run_id={run_id}")
    os.makedirs(folder, exist_ok=True)
    # date and run_id are in the folder names; a column of the same name would clash when reading
    df = df.drop(columns=[c for c in ("date", "run_id") if c in df.columns])
    # microseconds: Spark reads these timestamps as timestamp_ntz, like the tables it writes itself
    df.to_parquet(os.path.join(folder, f"{name}.parquet"), index=False,
                  coerce_timestamps="us", allow_truncated_timestamps=True)


def read_table(table, config, filters=None, columns=None):
    """A gold table as pandas (empty if it doesn't exist yet); filters on its date partitions."""
    path = table_path(table, config)
    # no table yet, or a folder without a file in it (e.g. emptied by hand): an empty frame
    if not any(name.endswith(".parquet") for _, _, names in os.walk(path) for name in names):
        return pd.DataFrame(columns=columns or [])
    return pd.read_parquet(path, filters=filters, columns=columns)


def task_runs_dir(run_id, config):
    return os.path.join(config["paths"]["datamart"], "ops", "task_runs", f"run_id={run_id}")


def record_task_run(record, config):
    """main.py writes one record per task run; log_run collects them into ops_run_log."""
    folder = task_runs_dir(record["run_id"], config)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{record['task']}.json"), "w") as f:
        json.dump(record, f, default=str)


def read_task_runs(run_id, config):
    folder = task_runs_dir(run_id, config)
    if not os.path.exists(folder):
        return []
    records = []
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name)) as f:
            records.append(json.load(f))
    return records