from datetime import datetime

from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

from pipeline_tasks import config, forecast_tasks

# each run forecasts the next hour: T = data_interval_end
with DAG(
    dag_id="hourly_forecast",
    schedule=CronDataIntervalTimetable("@hourly", timezone="UTC"),
    start_date=datetime.fromisoformat(config["deployment"]["go_live"]),
    end_date=datetime.fromisoformat(config["window"]["end"]),
    catchup=False,  # the go-live replay is started on purpose, with a backfill
    max_active_runs=config["schedules"]["max_active_runs"],
    tags=["live"],
) as dag:
    forecast_tasks()
