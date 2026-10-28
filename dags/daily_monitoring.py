from datetime import datetime

from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

from pipeline_tasks import config, monitoring_tasks

# the run for day [D, D+1) fires at the end of D and monitors day D-1, whose labels it builds first
with DAG(
    dag_id="daily_monitoring",
    schedule=CronDataIntervalTimetable("@daily", timezone="UTC"),
    start_date=datetime.fromisoformat(config["deployment"]["go_live"]),
    end_date=datetime.fromisoformat(config["window"]["end"]),
    catchup=False,  # started with the replay, with a backfill
    max_active_runs=1,  # the rolling checks read the earlier days' results
    tags=["live", "monitoring"],
) as dag:
    monitoring_tasks()
