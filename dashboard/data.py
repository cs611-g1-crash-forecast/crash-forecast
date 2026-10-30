"""What the planner page reads and works out, kept apart from Streamlit so the tests can call it.
The dashboard only reads: the forecasts and labels the pipeline writes to gold, and the area boundaries."""
import json
import os
from datetime import timedelta

import pandas as pd

from utils.tables import read_table, table_path

KEY = ["community_area", "hour_start"]


def forecast_dates(config):
    """The days with forecasts, oldest first, from the table's date partitions (no file is read)."""
    path = table_path("model_predictions", config)
    if not os.path.isdir(path):
        return []
    return sorted(name.removeprefix("date=") for name in os.listdir(path) if name.startswith("date="))


def production_forecasts(day, config):
    """The production forecasts for the hours of one day. Planners see production only: the shadow's
    forecasts are for monitoring. An hour forecast twice (a rerun) keeps its latest forecast."""
    rows = read_table("model_predictions", config, filters=[("date", "==", day), ("role", "==", "production")])
    if rows.empty:
        return rows
    return (rows.sort_values("predicted_at").drop_duplicates(KEY, keep="last")
            .drop(columns=["date", "run_id"], errors="ignore").reset_index(drop=True))


def labels(day, config):
    """The crash counts for the hours of one day, from the latest label snapshot (as notebook 11)."""
    rows = read_table("label_store", config, filters=[("date", "==", day)])
    if rows.empty:
        return rows
    return rows.sort_values("label_snapshot_date").drop_duplicates(KEY, keep="last").reset_index(drop=True)


def areas(config):
    """The 77 community areas: {number: (name, geometry)} from the boundaries file."""
    with open(os.path.join(config["paths"]["reference"], config["sources"]["boundaries"]["file"])) as f:
        features = json.load(f)["features"]
    return {int(a["properties"]["area_numbe"]): (a["properties"]["community"].title(), a["geometry"])
            for a in features}


def shift_start(hour, shift_hours):
    """The start of the shift an hour falls in: shifts start at midnight, every shift_hours hours."""
    return pd.Timestamp(hour).floor(f"{shift_hours}h")


def rank_areas(forecasts, hour, names):
    """The areas ranked by their forecast for one hour, highest first; ties go to the lower area number."""
    hour_rows = forecasts[forecasts["hour_start"] == hour].sort_values("community_area")
    ranked = hour_rows.sort_values("predicted_count", ascending=False, kind="stable")
    return pd.DataFrame({
        "rank": range(1, len(ranked) + 1),
        "community_area": ranked["community_area"].to_numpy(),
        "area": ranked["community_area"].map(names).to_numpy(),
        "forecast": ranked["predicted_count"].to_numpy(),
    })


def shift_result(forecasts, shift_labels, first, shift_hours, k):
    """How the shift went, once all its hours are labelled: crashes per area, and P0, the share of the
    shift's crashes in the k areas forecast highest at its start. None until then."""
    last = first + timedelta(hours=shift_hours)
    in_shift = shift_labels[(shift_labels["hour_start"] >= first) & (shift_labels["hour_start"] < last)]
    if in_shift.empty or in_shift["hour_start"].nunique() < shift_hours:
        return None
    actual = in_shift.groupby("community_area")["crash_count"].sum()
    staged = rank_areas(forecasts, first, {})["community_area"].head(k)
    total, caught = int(actual.sum()), int(actual.reindex(staged, fill_value=0).sum())
    return {"actual": actual, "total": total, "caught": caught, "p0": caught / total if total else None}
