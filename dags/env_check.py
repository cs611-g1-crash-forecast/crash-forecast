from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="env_check",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["setup"],
) as dag:
    BashOperator(
        task_id="spark_and_data",
        bash_command="python -m utils.env_check",
        cwd="/opt/airflow/crash-forecast",
    )