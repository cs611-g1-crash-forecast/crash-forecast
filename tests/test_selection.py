"""Tests for the selection rule: which trained candidate shadows production."""
import pytest

from utils.model import select_candidate

CONFIG = {"models": {"selection": {"margin": 0.01, "tie": 0.002}}}


def entry(algorithm, p0, deviance):
    return {"algorithm": algorithm, "model_version": algorithm,
            "validation_metrics": {"P0": p0, "poisson_deviance": deviance}}


@pytest.mark.parametrize("p0_m1, p0_m2, expected", [
    (0.300, 0.310, "M2"),    # the best P0 wins
    (0.300, 0.2855, "M1"),   # M2 is below B0 + margin, so it can't be chosen
])
def test_the_best_validation_p0_that_beats_b0_wins(p0_m1, p0_m2, expected):
    entries = [entry("B0", 0.276, 0.60), entry("M1", p0_m1, 0.56), entry("M2", p0_m2, 0.55)]
    assert select_candidate(entries, CONFIG)["algorithm"] == expected


def test_nothing_is_selected_when_no_candidate_beats_b0_by_the_margin():
    entries = [entry("B0", 0.276, 0.60), entry("M1", 0.280, 0.56), entry("M2", 0.285, 0.55)]
    assert select_candidate(entries, CONFIG) is None


def test_a_p0_tie_goes_to_the_lower_deviance():
    # M1 is 0.001 ahead on P0, within the tie, so M2's lower deviance decides
    entries = [entry("B0", 0.276, 0.60), entry("M1", 0.301, 0.56), entry("M2", 0.300, 0.54)]
    assert select_candidate(entries, CONFIG)["algorithm"] == "M2"
