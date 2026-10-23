from datetime import datetime

import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

with open("/opt/airflow/crash-forecast/config/pipeline.yaml") as f:
    config = yaml.safe_load(f)

SOURCES = ["crashes", "weather", "tracker", "taxi", "licences"]

# every task processes this run's hour, [data_interval_start, data_interval_end), and the features
# and forecasts are for the next hour T = data_interval_end
INTERVAL = "--start {{ data_interval_start.isoformat() }} --end {{ data_interval_end.isoformat() }}"


def task(task_id, command, **kwargs):
    # weight_rule="upstream": a task further down the run goes first, so when several hours queue
    # up (the replay, or a catch-up after an outage) the earliest hour's forecast is finished first
    return BashOperator(task_id=task_id, bash_command=f"python main.py {command} {INTERVAL}",
                        cwd="/opt/airflow/crash-forecast", weight_rule="upstream", **kwargs)


with DAG(
    dag_id="hourly_forecast",
    schedule=CronDataIntervalTimetable("@hourly", timezone="UTC"),
    start_date=datetime.fromisoformat(config["deployment"]["go_live"]),
    end_date=datetime.fromisoformat(config["window"]["end"]),
    catchup=False,  # the go-live replay is started on purpose, with a backfill
    max_active_runs=config["schedules"]["max_active_runs"],
    tags=["live"],
) as dag:
    features = task("gold_features", "gold_features", pool="spark",
                    depends_on_past=True)  # T's features read the earlier hours' silver
    for source in SOURCES:
        bronze = task(f"bronze_{source}", f"bronze --source {source}", pool="spark")
        silver = task(f"silver_{source}", f"silver --source {source}", pool="spark")
        bronze >> silver >> features

    dq = task("check_data_quality", "check_data_quality")
    infer_production = task("infer_production", "infer --role production")
    infer_shadow = task("infer_shadow", "infer --role shadow")
    # the run log is written whatever happened, so a failed run is logged too
    log = task("log_run", "log_run", trigger_rule="all_done")

    features >> dq >> [infer_production, infer_shadow] >> log