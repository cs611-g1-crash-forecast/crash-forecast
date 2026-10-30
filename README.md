# crash-forecast

Hourly forecasts of traffic-crash counts for each of Chicago's 77 community areas, so that EMS planners can stage ambulances before the hour starts. CS611 Machine Learning Engineering group project (Group 1).

Five public sources (crashes, O'Hare weather, the traffic tracker, taxi trips, business licences) go through bronze, silver and gold layers into a feature store. Three models are trained and registered: B0 (the area's average for that hour of the week), M1 (Poisson regression) and M2 (XGBoost, Poisson). From go-live, every hour the production model and a shadow forecast the next hour for every area, and a daily job monitors data quality, performance (P0: the share of a shift's crashes that fall in the 10 areas forecast highest), drift, attribution and fairness, then promotes, rolls back or retrains by rule. Everything runs in Airflow, in one Docker image.

## Run it on the full data

The study covers March 2018 to August 2023, with go-live on 1 January 2023. The raw data (about 20 GB) goes in `../data/`, next to this folder, one subfolder per source. You need Docker Desktop with at least 8 GB of memory, 12 GB or more if you can spare it (Settings → Resources), at least 30 GB of free disk besides the data (the image is about 6 GB, `datamart/` grows to about 13 GB), and a bash shell (macOS, Linux, or WSL on Windows). With under 12 GB, let one Spark job run at a time: after starting the stack, `docker compose exec airflow-scheduler airflow pools set spark 1 "Spark jobs at once"`.

In the repo, `config/pipeline.yaml` holds the full study's settings. In the submission zip it holds the sample's, so there, first run `cp config/pipeline.full.yaml config/pipeline.yaml`.

```sh
docker compose up -d --build     # first build: about 5 min
```

On Linux, first run `echo "AIRFLOW_UID=$(id -u)" > .env`, so that the files the containers write belong to you.

The pipeline runs in four stages, in the order the system runs:

| # | What runs | DAG | Full data | Sample |
|---|---|---|---|---|
| 1 | The bulk history: bronze, silver, the label store and the feature store, a month per run | `backfill_history` | March 2018 to December 2022 | September to December 2022 |
| 2 | The first models: B0, M1 and M2 trained, compared on validation, registered (B0 serves first; the best of M1 and M2 shadows it) | `model_training` | trained on April 2018 – December 2021, chosen on January – June 2022, tested on July – December 2022 | trained on October, chosen on November, tested on December |
| 3 | The simulated live feed: each source's records filed by the date they arrive | `main.py stage_feed` | 1 January – 31 August 2023 | 1–14 January 2023 |
| 4 | Go-live: each run forecasts a day's 24 hours with the production and shadow models, then monitors the day before and acts on it | `replay_days` | 1 January – 31 August 2023 | 1–14 January 2023 |

```sh
af() { docker compose exec airflow-scheduler airflow "$@"; }
af dags unpause backfill_history
af backfill create --dag-id backfill_history --from-date 2018-03-01 --to-date 2022-12-01 --reprocess-behavior completed
# when every run has succeeded:
af dags unpause model_training && af dags trigger model_training
for s in crashes weather tracker taxi licences; do
  docker compose exec jupyter python main.py stage_feed --source $s --start 2023-01-01T00:00 --end 2023-09-01T00:00
done
af dags unpause replay_days
af backfill create --dag-id replay_days --from-date 2023-01-01 --to-date 2023-08-31 --reprocess-behavior completed --max-active-runs 1
```

On a 14-CPU laptop with 11.7 GB for Docker, the history took about 1.5 hours (bronze 41 min, silver and gold about 50 min), training about 4 minutes and the 240 replayed days 11 hours. Keep other Docker projects stopped meanwhile: Spark needs the memory. `hourly_forecast` and `daily_monitoring` are the same tasks on an hourly and a daily schedule, as they would run in production; the replay uses `replay_days` so that eight months take hours, not a week.

Watch it in Airflow at http://localhost:8080 (no login): **DAGs → a DAG → Runs**, and a task's **Logs**. JupyterLab is at http://localhost:8888, and the planner dashboard at http://localhost:8501.

Stop the stack with `docker compose down`; add `-v` to also delete Airflow's database. To run again from scratch, also delete `datamart/` and `model_bank/`.

## Check it end to end on the sample

The full run takes about 13 hours, so the submission zip comes with a sample of the data that runs every stage in about 30 minutes (see [The sample](#the-sample)). The zip holds two folders side by side: `crash-forecast/` (this code, with the sample's settings as `config/pipeline.yaml`) and `data/` (the sample). It needs Docker Desktop with at least 8 GB of memory and about 15 GB of free disk.

```sh
cd crash-forecast
docker compose up -d --build     # first build: about 5 min
tools/run_sample.sh              # the four stages above, on the sample: about 30 min
```

`tools/run_sample.sh` drives Airflow through the four stages, waiting for each and stopping if a run fails. In the repo, the zip is built from a tag with `tools/package.sh <tag>` (it cuts the sample from `../data/`).

## Tests

Run by CI on every pull request, and by hand with the stack up:

```sh
docker compose exec jupyter python -m pytest tests/ -q -p no:cacheprovider
```

## What to look at

| Where | What |
|---|---|
| `datamart/gold/model_predictions/` | Every forecast: area, hour, model, role (production or shadow) |
| `datamart/gold/label_store/`, `feature_store/` | The crash counts per area-hour, and the 25 features the models read |
| `model_bank/registry.json` | Every trained model with its status (production, shadow, archived, rejected) and the reason for each change |
| `model_bank/<model_version>/` | The model artefact: the fitted model, its validation and test metrics, the training distributions for drift checks |
| `datamart/gold/monitoring/` | One table per check: `data_quality`, `model_performance`, `stability` (PSI, CSI), `attribution`, `actions`, and `ops_run_log` (every task of every run, with its time) |
| `notebooks/11_planner_view.ipynb` | The user's view: one shift's forecast on a map of the 77 areas, the 10 areas to stage in, and how many of the shift's crashes they caught |
| `notebooks/12_monitoring_report.ipynb` | The monitoring tables as charts over the replayed period |
| http://localhost:8501 | The planner dashboard: notebook 11's view, live. It reads the forecasts in gold as `hourly_forecast` or `replay_days` writes them; with "Follow the newest forecast" on, it moves on by itself every `dashboard.refresh_seconds` |

The notebooks are saved with the outputs of the full eight-month replay. To redraw them after a run on the sample, open them in JupyterLab and run all cells; in notebook 11, first set `DAY = "2023-01-06"` (a Friday in the sample's live weeks).

## The sample

The sample is cut from the full data by `tools/make_sample.py`, in the same folders, file formats and column names, so the pipeline runs on it unchanged:

- **Window:** September 2022 to 14 January 2023, with go-live on 1 January 2023: four months of history (September gives October its four weeks of history), then the first two live weeks (the drift checks compare a whole week, so the second week has them). The licences keep every licence issued before the window, since those can still be active in it.
- **Fewer columns:** taxi trips keep the two columns the pipeline reads (when the trip ended, the pickup area), and the tracker the seven it reads.
- **Fewer tracker readings:** the earliest reading of each road segment in each hour (about one in six). A month of tracker data is about 200 MB; this keeps the whole zip at about 27 MB.

`config/pipeline.sample.yaml` holds the sample's settings: its window, go-live, splits and file names; everything else is the full study's `config/pipeline.yaml`, and `tests/test_config.py` checks that the two differ only there. In the submission zip the sample's settings are `config/pipeline.yaml` and the full study's `config/pipeline.full.yaml`. Models trained on one month and validated on one are not the study's models, and two weeks are too short for a promotion (which needs 28 days of evidence), so the sample shows that every stage runs, not the study's results. Those are in the notebooks and the report. On the sample, M1 becomes the shadow, and the drift checks flag citywide taxi trips: January's are far below October's, the month the models trained on.

## Notebooks

| Notebook | What it checks or shows |
|---|---|
| `01_bronze_checks` | Bronze holds exactly the raw data, for every source |
| `02_silver_checks` | Silver's cleaning rules for crashes and weather did what the EDA said they should |
| `03_gold_checks` | The label store and feature store: the full area-hour grid, and no feature uses data that arrived after its hour |
| `04_forecast_checks` | The first live hours: the feed, the data-quality gate, production and shadow forecasts |
| `05_monitoring_checks` | The first monitored day: labels, P0, PSI, attribution, fairness |
| `06_model_experiments` | B0, M1 and M2 compared, with the COVID months kept or left out |
| `07_model_tuning` | M2's settings searched on the train split |
| `08_tracker_checks` | Tracker silver, coverage and the congestion features |
| `09_taxi_licence_checks` | Taxi and licence silver and their features |
| `10_integration_checks` | All six feature families in gold, and the models retrained on them |
| `11_planner_view` | One shift, as a planner sees it |
| `12_monitoring_report` | The go-live replay's monitoring, as charts |

## Layout

| Path | What it holds |
|---|---|
| `Dockerfile`, `requirements.txt` | One image for everything: the official Airflow image plus Java 17, PySpark and the ML libraries |
| `docker-compose.yaml` | Airflow (API server, scheduler, DAG processor, Postgres metadata database), JupyterLab and the dashboard |
| `config/` | Every date, path and threshold the code reads (`pipeline.yaml`), and the sample's settings |
| `main.py` | One pipeline stage for one interval: `python main.py <stage> --start … --end …` |
| `dags/` | The Airflow DAGs; `pipeline_tasks.py` defines the tasks the live DAGs share |
| `utils/` | The pipeline code, one module per stage |
| `dashboard/` | The Streamlit planner dashboard: a service of its own that reads gold and writes nothing |
| `tests/` | Unit tests: point-in-time features, the selection and action rules, the configs |
| `tools/` | `run_sample.sh`; `make_sample.py` and `package.sh`, which build the submission zip |
| `notebooks/` | Checks and reports, run in JupyterLab |
| `reference/` | The community-area boundaries (City of Chicago data portal) |
| `.github/workflows/ci.yml` | CI: the tests on every pull request into `main` |
| `data/` | Raw data, mounted read-only from `../data/` (not committed) |
| `datamart/`, `model_bank/`, `logs/` | What the pipeline writes (not committed) |
