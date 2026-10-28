from datetime import datetime

from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

from pipeline_tasks import config, forecast_tasks, monitoring_tasks

# the go-live replay between the hourly demo windows: each run forecasts a whole day's 24 hours,
# then monitors the day before, as hourly_forecast and daily_monitoring would by the end of that day.
# One run at a time, so a promotion or rollback reaches the next day's forecasts. Never replay the
# same days as hourly runs: the two would write the same hours twice
with DAG(
    dag_id="replay_days",
    schedule=CronDataIntervalTimetable("@daily", timezone="UTC"),
    start_date=datetime.fromisoformat(config["deployment"]["go_live"]),
    end_date=datetime.fromisoformat(config["window"]["end"]),
    catchup=False,  # run on purpose, with a backfill
    max_active_runs=1,
    tags=["live", "replay"],
) as dag:
    forecasts, monitors = forecast_tasks(), monitoring_tasks()
    # monitoring waits for both forecasts: if they weren't made, the run fails instead of ending green
    for forecast in forecasts:
        forecast >> monitors
    # the run log waits for the monitoring too, so a replayed day's log has every task in it
    dag.get_task("apply_actions") >> dag.get_task("log_run")
