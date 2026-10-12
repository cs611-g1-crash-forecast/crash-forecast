from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

SOURCES = ["crashes", "weather", "tracker", "taxi", "licences", "boundaries"]
SILVER_SOURCES = ["crashes", "weather"]  # tracker, taxi and licences get silver in phase 3a

# every task runs one pipeline stage for this run's month: [data_interval_start, data_interval_end)
INTERVAL = "--start {{ data_interval_start.isoformat() }} --end {{ data_interval_end.isoformat() }}"

with DAG(
    dag_id="backfill_history",
    # a run covers one month, [start, end); in Airflow 3 a plain "@monthly" has no interval
    schedule=CronDataIntervalTimetable("@monthly", timezone="UTC"),
    start_date=datetime(2018, 3, 1),
    end_date=datetime(2022, 12, 1),  # last month before go-live (2023-01-01)
    catchup=False,  # history is loaded on purpose, with a backfill
    tags=["history"],
) as dag:
    bronze = {}
    for source in SOURCES:
        bronze[source] = BashOperator(
            task_id=f"bronze_{source}",
            bash_command=f"python main.py bronze --source {source} {INTERVAL}",
            cwd="/opt/airflow/crash-forecast",
            pool="spark",  # at most 2 Spark jobs at once, to fit in Docker's memory
        )

    for source in SILVER_SOURCES:
        silver = BashOperator(
            task_id=f"silver_{source}",
            bash_command=f"python main.py silver --source {source} {INTERVAL}",
            cwd="/opt/airflow/crash-forecast",
            pool="spark",
        )
        # silver reads the batch its bronze task wrote; crashes also need the boundaries in bronze
        bronze[source] >> silver
        if source == "crashes":
            bronze["boundaries"] >> silver
