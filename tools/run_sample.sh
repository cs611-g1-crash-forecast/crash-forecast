#!/usr/bin/env bash
# Runs the pipeline end to end on the data sample, through Airflow, in the order the system runs:
# the bulk history (September to December 2022), the first model, the simulated live feed, then the
# first two weeks after go-live, a day per run: every hour forecast, and each day monitored the day after.
# Start the stack first (docker compose up -d --build), then run this from the crash-forecast folder.
set -euo pipefail

af() { docker compose exec -T airflow-scheduler airflow "$@"; }

# wait until the DAG has no run queued or running; stop if one failed
wait_for() {
  sleep 30
  while af dags list-runs "$1" 2>/dev/null | grep -q -E "running|queued"; do sleep 30; done
  if af dags list-runs "$1" 2>/dev/null | grep -q failed; then
    echo "$1: a run failed. Open it in the Airflow UI (http://localhost:8080) to read the task's log."
    exit 1
  fi
  echo "$1: done"
}

echo "Waiting for Airflow to load the DAGs ..."
until af dags list 2>/dev/null | grep -q replay_days; do sleep 10; done

# two Spark jobs at once need about 11 GB for Docker; with less, one at a time
if [ "$(docker info --format '{{.MemTotal}}')" -lt 11000000000 ]; then
  af pools set spark 1 "Spark jobs at once" > /dev/null
fi

echo "1/4 Bulk history: one backfill_history run per month, September to December 2022"
af dags unpause backfill_history > /dev/null
af backfill create --dag-id backfill_history --from-date 2022-09-01 --to-date 2022-12-01 --reprocess-behavior completed > /dev/null
wait_for backfill_history

echo "2/4 The first model: B0, M1 and M2 trained on the history and registered"
af dags unpause model_training > /dev/null
af dags trigger model_training > /dev/null
wait_for model_training

echo "3/4 The simulated live feed: what arrives from 1 to 14 January 2023, by arrival date"
for source in crashes weather tracker taxi licences; do
  docker compose exec -T jupyter python main.py stage_feed --source "$source" \
    --start 2023-01-01T00:00 --end 2023-01-15T00:00 2>/dev/null | grep "^stage_feed"
done

echo "4/4 The live weeks: a day per run, 1 to 14 January 2023"
af dags unpause replay_days > /dev/null
af backfill create --dag-id replay_days --from-date 2023-01-01 --to-date 2023-01-14 \
  --reprocess-behavior completed --max-active-runs 1 > /dev/null
wait_for replay_days

echo "Done. The results are in datamart/gold/ and model_bank/; see the README's \"What to look at\"."
