from datetime import datetime

import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with open("/opt/airflow/crash-forecast/config/pipeline.yaml") as f:
    config = yaml.safe_load(f)

# the first model trains on everything before go-live; the splits inside that window come from config
WINDOW = f"--start {config['window']['start']} --end {config['deployment']['go_live']}"

with DAG(
    dag_id="model_training",
    schedule=None,  # triggered: by hand for the first model; later by decide_actions for a retrain
    start_date=datetime(2018, 3, 1),
    catchup=False,
    tags=["training"],
) as dag:
    # one task: load the stores, fit, evaluate, SHAP baseline, register
    train = BashOperator(
        task_id="train",
        bash_command=f"python main.py train {WINDOW}",
        cwd="/opt/airflow/crash-forecast",
        pool="spark",  # not Spark, but as heavy on memory, so it shares the pool
    )