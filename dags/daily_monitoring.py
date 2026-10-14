from datetime import datetime

import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

with open("/opt/airflow/crash-forecast/config/pipeline.yaml") as f:
    config = yaml.safe_load(f)

# the run for day [D, D+1) fires at the end of D and monitors day D-1, whose labels it builds first
INTERVAL = "--start {{ data_interval_start.isoformat() }} --end {{ data_interval_end.isoformat() }}"


def task(task_id, command, **kwargs):
    return BashOperator(task_id=task_id, bash_command=f"python main.py {command} {INTERVAL}",
                        cwd="/opt/airflow/crash-forecast", **kwargs)


with DAG(
    dag_id="daily_monitoring",
    schedule=CronDataIntervalTimetable("@daily", timezone="UTC"),
    start_date=datetime.fromisoformat(config["deployment"]["go_live"]),
    end_date=datetime.fromisoformat(config["window"]["end"]),
    catchup=False,  # started with the replay, with a backfill
    max_active_runs=1,  # the rolling checks read the earlier days' results
    tags=["live", "monitoring"],
) as dag:
    labels = task("gold_labels", "gold_labels", pool="spark")
    performance = task("monitor_performance", "monitor_performance")
    fairness = task("monitor_fairness", "monitor_fairness")
    stability = task("monitor_stability", "monitor_stability")
    attribution = task("monitor_attribution", "monitor_attribution")
    actions = task("decide_actions", "decide_actions")

    labels >> [performance, fairness, stability]
    [performance, fairness, stability, attribution] >> actions
