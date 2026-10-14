from datetime import datetime

import pandas as pd

from utils.bronze import run_id_for
from utils.gold import target_hours
from utils.model import load_artefact, model_with_status, predict, read_gold, utc_now
from utils.tables import read_table, read_task_runs, write_rows


def read_features(start, end, config):
    """The feature rows for the target hours of the run [start, end): (start, end]."""
    targets = target_hours(start, end, config)
    first, last = targets.min().to_pydatetime(), targets.max().to_pydatetime()
    return read_gold("feature_store", None, first, last + pd.Timedelta(hours=1), config), targets


def check_data_quality(start, end, config):
    """The gate before inference: a failed check stops the run, the last good forecast stands."""
    run_id = run_id_for(start, end)
    features, targets = read_features(start, end, config)
    expected_rows = 77 * len(targets)
    checks = [
        # (layer, source, check, value, threshold, failed status, is the value bad?)
        ("gold", "feature_store", "rows", len(features), expected_rows, "alert", len(features) != expected_rows),
        ("gold", "feature_store", "duplicate_keys",
         int(features.duplicated(["community_area", "hour_start"]).sum()), 0, "alert", None),
        ("gold", "feature_store", "cutoff_not_hour_start",
         int((features["feature_cutoff"] != features["hour_start"]).sum()), 0, "alert", None),
        # a stale weather reading still allows a forecast, so only a warning
        ("gold", "feature_store", "weather_reading_age_hours_max",
         float(features["wx_age_hours"].max()), 3, "warn", None),
    ]
    rows = []
    for layer, source, check, value, threshold, failed, bad in checks:
        if bad is None:
            bad = value > threshold
        rows.append({"run_id": run_id, "period_start": start, "period_end": end, "layer": layer,
                     "source": source, "check": check, "value": float(value), "threshold": float(threshold),
                     "status": failed if bad else "ok"})
    dq = pd.DataFrame(rows)
    write_rows(dq, "monitoring/data_quality", start.date(), run_id, "check_data_quality", config)
    for r in rows:
        print(f"data quality {run_id}: {r['check']} = {r['value']:g} (threshold {r['threshold']:g}): {r['status']}")
    if (dq["status"] == "alert").any():
        raise SystemExit("data quality alert: inference is skipped for this run")
    return len(rows)


def infer(start, end, role, config):
    """Forecast the run's target hours with the model that has this role in the registry."""
    run_id = run_id_for(start, end)
    entry = model_with_status(role, config)
    if entry is None:
        print(f"infer {role} {run_id}: no {role} model in the registry, nothing to do")
        return 0
    artefact = load_artefact(entry)
    features, targets = read_features(start, end, config)
    predictions = pd.DataFrame({
        "community_area": features["community_area"].to_numpy(),
        "hour_start": features["hour_start"].to_numpy(),
        "model_version": entry["model_version"],
        "role": role,
        "predicted_count": predict(artefact, features),
        "feature_cutoff": features["feature_cutoff"].to_numpy(),
        "predicted_at": utc_now(),  # wall clock, UTC
    })
    # save the forecasts - IRL the table the planners' map reads
    for date, rows in predictions.groupby(predictions["hour_start"].dt.date):
        write_rows(rows, "model_predictions", date, run_id, role, config)
    print(f"infer {role} {run_id}: {entry['model_version']}, {len(predictions)} forecasts for "
          f"{targets.min():%Y-%m-%d %H:%M} to {targets.max():%Y-%m-%d %H:%M}, "
          f"citywide {predictions['predicted_count'].sum():.2f} crashes expected")
    return len(predictions)


def log_run(start, end, config):
    """ops_run_log: one row per task of this run, and whether the forecast was fresh,
    i.e. the production forecast for every target hour was written by the run that ends at that hour."""
    run_id = run_id_for(start, end)
    targets = target_hours(start, end, config)
    predictions = read_table("model_predictions", config,
                             filters=[("run_id", "==", run_id), ("role", "==", "production")])
    fresh = len(predictions) == 77 * len(targets)
    dag = "hourly_forecast" if run_id.startswith(("h_", "d_")) else "backfill_history"
    rows = [{"run_id": run_id, "dag": dag, "task": r["task"], "interval_start": start, "interval_end": end,
             "started_at": datetime.fromisoformat(r["started_at"]), "seconds": r["seconds"],
             "rows_in": r.get("rows_in"), "rows_out": r["rows_out"], "status": r["status"],
             "forecast_fresh": fresh}
            for r in read_task_runs(run_id, config) if r["task"] != "log_run"]
    log = pd.DataFrame(rows).astype({"rows_in": "float", "rows_out": "float"})
    write_rows(log, "monitoring/ops_run_log", start.date(), run_id, "log_run", config)
    print(f"log_run {run_id}: {len(rows)} tasks, {log['seconds'].sum():.0f} s in tasks, "
          f"failed: {(log['status'] != 'success').sum()}, forecast fresh: {fresh}")
    return len(rows)
