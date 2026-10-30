# crash-forecast

Hourly forecasts of traffic-crash counts for each of Chicago's 77 community areas, so that EMS planners can stage ambulances before the hour starts. CS611 Machine Learning Engineering group project (Group 1).

## Run it

You need Docker Desktop with at least 8 GB of memory (Settings → Resources). The raw data is not in this repo: put the five source folders in a `data/` folder **next to** the repo (`../data/`), which is mounted into the containers read-only.

```sh
docker compose up -d --build
```

The first build takes several minutes. Then:

| Service | Address |
|---|---|
| Airflow | http://localhost:8080 |
| JupyterLab | http://localhost:8888 |
| Planner dashboard | http://localhost:8501 |

The dashboard reads the forecasts in gold and fills in as `hourly_forecast` or `replay_days` writes them; with "Follow the newest forecast" on, it moves on by itself every `dashboard.refresh_seconds`.

To check the environment, open Airflow, find the `env_check` DAG, unpause it and trigger it. Its `spark_and_data` task log lists the versions, a small Spark job's result and the data folders it can see.

On Linux, first run `echo "AIRFLOW_UID=$(id -u)" > .env`, so that files written by the containers belong to you.

Stop the stack with `docker compose down`. Add `-v` to also delete Airflow's database.

## Layout

| Path | What it holds |
|---|---|
| `Dockerfile`, `requirements.txt` | One image for everything: the official Airflow image plus Java 17, PySpark and the ML libraries |
| `docker-compose.yaml` | Airflow (API server, scheduler, DAG processor, Postgres metadata database) and JupyterLab |
| `dags/` | Airflow DAGs |
| `utils/` | Pipeline code, one module per stage |
| `dashboard/` | The Streamlit planner dashboard: a service of its own that reads gold and writes nothing |
| `data/` | Raw data, mounted read-only from `../data/` (not committed) |
| `datamart/` | Pipeline output: bronze, silver and gold (not committed) |
