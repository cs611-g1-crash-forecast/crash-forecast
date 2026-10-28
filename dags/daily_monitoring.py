import os
from datetime import datetime, timedelta

import pandas as pd
import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import ShortCircuitOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

with open("/opt/airflow/crash-forecast/config/pipeline.yaml") as f:
    config = yaml.safe_load(f)

# the run for day [D, D+1) fires at the end of D and monitors day D-1, whose labels it builds first
INTERVAL = "--start {{ data_interval_start.isoformat() }} --end {{ data_interval_end.isoformat() }}"


def task(task_id, command, **kwargs):
    return BashOperator(task_id=task_id, bash_command=f"python main.py {command} {INTERVAL}",
                        cwd="/opt/airflow/crash-forecast", **kwargs)


def retrain_requested(data_interval_start):
    """True if decide_actions asked for a retrain on the labelled day; False skips the trigger."""
    day = (data_interval_start - timedelta(days=config["monitoring"]["label_lag_days"])).date()
    folder = f"/opt/airflow/crash-forecast/{config['paths']['datamart']}/gold/monitoring/actions/date={day}"
    return os.path.exists(folder) and (pd.read_parquet(folder)["action"] == "retrain").any()


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
    apply = task("apply_actions", "apply_actions")
    retrain_needed = ShortCircuitOperator(task_id="retrain_needed", python_callable=retrain_requested)
    # the retrain trains up to the first day of the labelled day's month (the labelled day is the
    # interval's start minus label_lag_days, 1): whole months only
    retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="model_training",
        conf={"end": "{{ (data_interval_start - macros.timedelta(days=1)).strftime('%Y-%m-01T00:00') }}"},
        # wait for the training: the next run then forecasts with the new shadow, and the training
        # never runs beside the next run's Spark jobs (together they ran out of memory at 11.7 GB)
        wait_for_completion=True,
        poke_interval=30,
    )

    labels >> [performance, fairness, stability]
    [performance, fairness, stability, attribution] >> actions >> apply >> retrain_needed >> retrain
