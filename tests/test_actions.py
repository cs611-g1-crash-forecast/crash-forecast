"""Tests for promotion and rollback: when the registry swaps production and shadow."""
from utils.monitor import apply_swap, swap_decision

CONFIG = {"monitoring": {"promote_margin": 0.005}}


def entry(model_version, status, reason="trained"):
    return {"model_version": model_version, "status": status,
            "status_history": [{"status": status, "at": "2023-01-01T00:00:00", "reason": reason}]}


def test_no_swap_without_a_full_window_for_both_models():
    registry = [entry("B0", "production"), entry("M2", "shadow")]
    assert swap_decision(registry, {("B0", "production"): 0.27}, CONFIG) is None


def test_the_shadow_is_promoted_when_ahead_by_more_than_the_margin():
    registry = [entry("B0", "production"), entry("M2", "shadow")]
    decision = swap_decision(registry, {("B0", "production"): 0.27, ("M2", "shadow"): 0.30}, CONFIG)
    assert decision["action"] == "promote"
    apply_swap(registry, decision, "2023-02-01T00:00:00")
    assert [(e["model_version"], e["status"]) for e in registry] == [("B0", "shadow"), ("M2", "production")]


def test_no_promotion_within_the_margin():
    registry = [entry("B0", "production"), entry("M2", "shadow")]
    assert swap_decision(registry, {("B0", "production"): 0.300, ("M2", "shadow"): 0.303}, CONFIG) is None


def test_a_promoted_model_behind_the_one_it_replaced_is_rolled_back():
    registry = [entry("B0", "shadow", "demoted by M2: ..."), entry("M2", "production", "promoted over B0: ...")]
    decision = swap_decision(registry, {("M2", "production"): 0.29, ("B0", "shadow"): 0.30}, CONFIG)
    assert decision["action"] == "rollback"
    apply_swap(registry, decision, "2023-03-01T00:00:00")
    assert [(e["model_version"], e["status"]) for e in registry] == [("B0", "production"), ("M2", "shadow")]
