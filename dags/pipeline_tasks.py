"""The tasks the DAGs share, for one interval [data_interval_start, data_interval_end):
the forecasting pipeline, and the daily monitoring with its actions. Not a DAG itself."""
import os
from datetime import timedelta

import pandas as pd
import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import ShortCircuitOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

with open("/opt/airflow/crash-forecast/config/pipeline.yaml") as f:
    config = yaml.safe_load(f)

SOURCES = ["crashes", "weather", "tracker", "taxi", "licences"]

# every task processes this run's interval; the features and forecasts are for the hours T in
# (start, end]: one hour for an hourly run, 24 for a daily one
INTERVAL = "--start {{ data_interval_start.isoformat() }} --end {{ data_interval_end.isoformat() }}"


def task(task_id, command, **kwargs):
    # weight_rule="upstream": a task further down the run goes first, so when several intervals queue
    # up (the replay, or a catch-up after an outage) the earliest one is finished first.
    # retries: a Spark task killed when the machine runs short of memory runs again 5 minutes later
    # instead of stopping the run (every stage rewrites its own partition, so a rerun is safe)
    return BashOperator(task_id=task_id, bash_command=f"python main.py {command} {INTERVAL}",
                        cwd="/opt/airflow/crash-forecast", weight_rule="upstream",
                        retries=2, retry_delay=timedelta(minutes=5), **kwargs)


def forecast_tasks():
    """Bronze and silver for every source, gold features, the data-quality gate, production and
    shadow inference, the run log. Returns the two inference tasks: what comes after the forecast
    should run only if it was made (the run log runs either way)."""
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
    return [infer_production, infer_shadow]


def retrain_requested(data_interval_start):
    """True if decide_actions asked for a retrain on the labelled day; False skips the trigger."""
    day = (data_interval_start - timedelta(days=config["monitoring"]["label_lag_days"])).date()
    folder = f"/opt/airflow/crash-forecast/{config['paths']['datamart']}/gold/monitoring/actions/date={day}"
    return os.path.exists(folder) and (pd.read_parquet(folder)["action"] == "retrain").any()


def monitoring_tasks():
    """The labels of the day before, the four monitors, decide_actions, apply_actions, and the retrain
    trigger when one was asked for. Returns the first tasks."""
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
    return [labels, attribution]
