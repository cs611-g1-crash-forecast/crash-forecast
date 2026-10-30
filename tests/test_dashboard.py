"""Tests for the planner dashboard's data helpers: the shift, the ranking and the shift's P0."""
from datetime import timedelta

import pandas as pd

from dashboard.data import rank_areas, shift_result, shift_start

FIRST = pd.Timestamp("2023-06-16 16:00")


def forecasts(hour, counts):
    return pd.DataFrame({"community_area": list(counts), "hour_start": hour, "predicted_count": list(counts.values())})


def shift_labels(first, hours, counts):
    """Every hour of the shift labelled; counts[area] crashes in its first hour, none after."""
    hour_starts = pd.date_range(first, periods=hours, freq="h")
    rows = [{"community_area": area, "hour_start": hour_start, "crash_count": n if hour_start == first else 0}
            for hour_start in hour_starts for area, n in counts.items()]
    return pd.DataFrame(rows)


def test_shifts_start_at_midnight_every_shift_hours():
    assert shift_start(pd.Timestamp("2023-06-16 16:00"), 8) == FIRST
    assert shift_start(pd.Timestamp("2023-06-16 23:00"), 8) == FIRST
    assert shift_start(pd.Timestamp("2023-06-16 07:00"), 8) == pd.Timestamp("2023-06-16 00:00")


def test_areas_rank_by_forecast_and_ties_go_to_the_lower_area_number():
    ranked = rank_areas(forecasts(FIRST, {5: 0.4, 2: 0.9, 9: 0.4, 1: 0.1}), FIRST, {2: "Two"})
    assert ranked["community_area"].tolist() == [2, 5, 9, 1]
    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    assert ranked["area"].iloc[0] == "Two"


def test_p0_is_the_share_of_the_shifts_crashes_in_the_areas_staged_at_its_start():
    # staged at the start (k = 2): areas 2 and 5; a later hour ranking area 9 first changes nothing
    later = FIRST + timedelta(hours=1)
    shift = pd.DataFrame({"community_area": [2, 5, 9, 2, 5, 9], "hour_start": [FIRST] * 3 + [later] * 3,
                          "predicted_count": [0.9, 0.5, 0.1, 0.1, 0.1, 0.9]})
    result = shift_result(shift, shift_labels(FIRST, 8, {2: 3, 5: 1, 9: 4}), FIRST, 8, 2)
    assert (result["total"], result["caught"]) == (8, 4)
    assert result["p0"] == 0.5


def test_no_result_until_every_hour_of_the_shift_is_labelled():
    labels = shift_labels(FIRST, 8, {2: 3})
    partly = labels[labels["hour_start"] < FIRST + timedelta(hours=5)]
    assert shift_result(forecasts(FIRST, {2: 0.9}), partly, FIRST, 8, 1) is None


def test_a_shift_with_no_crash_has_no_p0():
    result = shift_result(forecasts(FIRST, {2: 0.9}), shift_labels(FIRST, 8, {2: 0}), FIRST, 8, 1)
    assert result["total"] == 0 and result["p0"] is None
