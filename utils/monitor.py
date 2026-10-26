import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.bronze import run_id_for
from utils.gold import label_hours
from utils.model import (LABEL, load_artefact, metrics, psi, read_gold, read_labels, read_registry,
                         set_status, shap_values, utc_now, write_registry)
from utils.tables import read_table, table_path, write_rows

METRICS = ["P0", "MAE", "RMSE", "poisson_deviance", "bias"]


def labelled_day(start, end, config):
    """The daily run for [D, D+1) monitors day D-1: the day gold_labels has just labelled."""
    first, end_hour, _ = label_hours(start, end, config)
    return first, end_hour


def scored_between(first, last, config):
    """Every forecast for the hours [first, last) (both roles) with its label, and the labels' snapshot date."""
    predictions = read_table("model_predictions", config,
                             filters=[("date", ">=", str(first.date())), ("date", "<", str(last.date()))])
    predictions = predictions[(predictions["hour_start"] >= first) & (predictions["hour_start"] < last)]
    labels = read_labels(first, last, config)
    snapshot = read_gold("label_store", ["hour_start", "label_snapshot_date"], first, last, config)["label_snapshot_date"].max()
    rows = predictions.merge(labels, on=["community_area", "hour_start"], how="inner")
    return rows.rename(columns={LABEL: "actual", "predicted_count": "predicted"}), snapshot


def scored_rows(start, end, config):
    """Every forecast of the labelled day (both roles) with its label."""
    return scored_between(*labelled_day(start, end, config), config)


def period(start, end, config):
    first, last = labelled_day(start, end, config)
    return {"period_start": first, "period_end": last, "granularity": "day"}


def models_scored(rows):
    """The (model_version, role) pairs that forecast these rows."""
    return rows[["model_version", "role"]].drop_duplicates().itertuples(index=False)


def performance_rows(rows, when, snapshot, config):
    """P0 and the error metrics, citywide, for each model in each role it forecast in."""
    out = []
    for model_version, role in models_scored(rows):
        m = metrics(rows[(rows["model_version"] == model_version) & (rows["role"] == role)], config, {})
        for metric in METRICS:
            out.append({**when, "model_version": model_version, "role": role, "segment": "citywide",
                        "metric": metric, "value": float(m[metric]), "label_snapshot_date": snapshot})
    return out


def monitor_performance(start, end, config):
    """Model performance on the labelled day, on the rolling windows that end with it, and on the
    month before on the re-score day; per model and role. A rolling row is written only when the
    model forecast every hour of the window in that role, so it always means the whole window."""
    mon = config["monitoring"]
    first, last = labelled_day(start, end, config)
    rows, snapshot = scored_rows(start, end, config)
    out = performance_rows(rows, period(start, end, config), snapshot, config)
    go_live = datetime.fromisoformat(config["deployment"]["go_live"])
    for days in mon["rolling_days"]:
        if last - timedelta(days=days) < go_live:
            continue  # the window starts before go-live, so no model forecast all of it
        window, snapshot = scored_between(last - timedelta(days=days), last, config)
        when = {"period_start": last - timedelta(days=days), "period_end": last, "granularity": f"rolling_{days}d"}
        # every hour, not every date: a day-sized run also forecasts the next day's 00:00 hour
        hours_forecast = window.groupby(["model_version", "role"])["hour_start"].transform("nunique")
        out += performance_rows(window[hours_forecast == window["hour_start"].nunique()], when, snapshot, config)
    if first.day == mon["rescore_day"]:
        # the month before, now that most of its late reports are in (OOT1, OOT2, ... for the report)
        month_end = first.replace(day=1)
        month_start = month_end - relativedelta(months=1)
        window, snapshot = scored_between(month_start, month_end, config)
        when = {"period_start": month_start, "period_end": month_end, "granularity": "month"}
        out += performance_rows(window, when, snapshot, config)
    return write_monitoring("model_performance", out, start, end, "monitor_performance", config)


def monitor_fairness(start, end, config):
    """Bias drift: P0 and MAE per fairness group, with the groups the model was trained with."""
    rows, snapshot = scored_rows(start, end, config)
    out = []
    for model_version, role in models_scored(rows):
        entry = next(e for e in read_registry(config) if e["model_version"] == model_version)
        groups = load_artefact(entry)["fairness_groups"]
        m = metrics(rows[rows["model_version"] == model_version], config, groups)
        for group, values in m["groups"].items():
            for metric, value in values.items():
                out.append({**period(start, end, config), "model_version": model_version, "role": role,
                            "segment": group, "metric": metric, "value": float(value),
                            "label_snapshot_date": snapshot})
    return write_monitoring("model_performance", out, start, end, "monitor_fairness", config)


def stability_status(value, config):
    thresholds = config["monitoring"]["psi"]
    return "alert" if value > thresholds["alert"] else "warn" if value > thresholds["warn"] else "ok"


def monitor_stability(start, end, config):
    """Stability over the week that ends with the labelled day: PSI of each model's forecasts, and CSI of
    each feature it uses, against their training distributions. Written once a model has forecast every
    hour of the week: one day's hours are too few, and too alike, to compare with years of training.
    The calendar features are left out of CSI: they follow the clock, so a week can never spread like
    the training years. A model trained before the feature references existed has PSI only."""
    mon = config["monitoring"]
    _, last = labelled_day(start, end, config)
    days = mon["stability_days"]
    window_start = last - timedelta(days=days)
    if window_start < datetime.fromisoformat(config["deployment"]["go_live"]):
        # the first week after go-live: no model has forecast a whole week yet
        return write_monitoring("stability", [], start, end, "monitor_stability", config)
    rows, _ = scored_between(window_start, last, config)
    hours_forecast = rows.groupby("model_version")["hour_start"].transform("nunique")
    rows = rows[hours_forecast == rows["hour_start"].nunique()]
    common = {"period_start": window_start, "period_end": last, "granularity": f"rolling_{days}d"}
    features = None
    out = []
    for model_version in rows["model_version"].unique():
        entry = next(e for e in read_registry(config) if e["model_version"] == model_version)
        reference = load_artefact(entry)["stability_reference"]
        value = psi(rows.loc[rows["model_version"] == model_version, "predicted"], reference["prediction"])
        out.append({**common, "model_version": model_version, "variable": "prediction", "index": "PSI",
                    "value": value, "status": stability_status(value, config)})
        for feature, feature_reference in reference.get("features", {}).items():
            if feature.split("_")[0] in mon["csi_skip_families"]:
                continue
            if features is None:
                features = read_gold("feature_store", None, window_start, last, config)
            values = features[feature].dropna()  # a missing value has no bin, as in training
            if values.empty:
                continue
            value = psi(values.to_numpy(), feature_reference)
            out.append({**common, "model_version": model_version, "variable": feature, "index": "CSI",
                        "value": value, "status": stability_status(value, config)})
    return write_monitoring("stability", out, start, end, "monitor_stability", config)


def ndcg(live, baseline):
    """How well the live ranking of feature attributions keeps the training ranking (SageMaker Clarify):
    DCG of the live order, scored with the baseline attributions, over the DCG of the baseline order."""
    discounts = 1 / np.log2(np.arange(2, len(baseline) + 2))
    live_order = sorted(live, key=live.get, reverse=True)
    ideal_order = sorted(baseline, key=baseline.get, reverse=True)
    dcg = sum(baseline[f] * d for f, d in zip(live_order, discounts))
    ideal = sum(baseline[f] * d for f, d in zip(ideal_order, discounts))
    return dcg / ideal if ideal > 0 else 1.0


def monitor_attribution(start, end, config):
    """Feature-attribution drift: mean |SHAP| per feature on the labelled day, ranked,
    and the nDCG of that ranking against the training baseline. Needs no labels."""
    first, last = labelled_day(start, end, config)
    out = []
    for entry in read_registry(config):
        if entry["status"] not in ("production", "shadow"):
            continue
        artefact = load_artefact(entry)
        X = read_gold("feature_store", None, first, last, config)
        live = shap_values(artefact, X).abs().mean().to_dict()
        score = ndcg(live, artefact["shap_baseline"])
        ranked = sorted(live, key=live.get, reverse=True)
        common = {**period(start, end, config), "model_version": entry["model_version"], "method": "shap"}
        for rank, feature in enumerate(ranked, start=1):
            out.append({**common, "feature": feature, "value": float(live[feature]), "rank": rank, "status": "ok"})
        status = "alert" if score < config["monitoring"]["attribution_ndcg"] else "ok"
        out.append({**common, "feature": "__ndcg__", "value": float(score), "rank": 0, "status": status})
    return write_monitoring("attribution", out, start, end, "monitor_attribution", config)


def write_monitoring(table, out, start, end, task, config):
    first, _ = labelled_day(start, end, config)
    if not out:
        # e.g. stability in the first week: no file, rather than one without columns. A file that an
        # earlier run of this interval wrote is removed, so a rerun still replaces exactly what it wrote
        stale = os.path.join(table_path(f"monitoring/{table}", config), f"date={first.date()}",
                             f"run_id={run_id_for(start, end)}", f"{task}.parquet")
        if os.path.exists(stale):
            os.remove(stale)
        print(f"{task} {first:%Y-%m-%d}: nothing to write yet")
        return 0
    write_rows(pd.DataFrame(out), f"monitoring/{table}", first.date(), run_id_for(start, end), task, config)
    for r in out:
        name = r.get("metric") or r.get("index") or r.get("feature")
        if r.get("index") == "CSI" and r["status"] == "ok":
            continue  # one line per feature would bury the rest; the table has them all
        parts = [r["model_version"], r.get("role"), r.get("segment"), r.get("granularity"),
                 r.get("variable") if r.get("index") == "CSI" else None, f"{name} = {r['value']:.4f}", r.get("status")]
        print(f"{task} {first:%Y-%m-%d}: " + " ".join(p for p in parts if p))
    return len(out)


# ---------- thresholds and actions ----------

def swap_decision(registry, rolling_p0, config):
    """Promote or roll back, from each model's rolling P0 in its current role.
    rolling_p0: {(model_version, role): P0} over the promotion window, for the models that forecast
    every hour of it in that role. Both models need a full window, so every swap has a window of
    evidence, and the window starts again after a swap.
    - rollback: the production model was promoted over the current shadow, and the shadow is ahead;
    - promote: otherwise, the shadow is ahead of production by more than the margin."""
    production = next((e for e in registry if e["status"] == "production"), None)
    shadow = next((e for e in registry if e["status"] == "shadow"), None)
    if production is None or shadow is None:
        return None
    p0_production = rolling_p0.get((production["model_version"], "production"))
    p0_shadow = rolling_p0.get((shadow["model_version"], "shadow"))
    if p0_production is None or p0_shadow is None:
        return None
    evidence = f"rolling P0 production {p0_production:.3f}, shadow {p0_shadow:.3f}"
    promoted_over_shadow = production["status_history"][-1]["reason"].startswith(f"promoted over {shadow['model_version']}")
    if promoted_over_shadow:
        if p0_shadow > p0_production:
            return {"action": "rollback", "model_version": shadow["model_version"],
                    "reason": f"{production['model_version']} is behind the model it replaced", "evidence": evidence}
        return None
    if p0_shadow - p0_production > config["monitoring"]["promote_margin"]:
        return {"action": "promote", "model_version": shadow["model_version"],
                "reason": f"shadow ahead of {production['model_version']} by more than the margin", "evidence": evidence}
    return None


def apply_swap(registry, decision, at):
    """Swap production and shadow in the registry for a promote or rollback decision."""
    production = next(e for e in registry if e["status"] == "production")
    shadow = next(e for e in registry if e["status"] == "shadow")
    if decision["action"] == "promote":
        set_status(shadow, "production", f"promoted over {production['model_version']}: {decision['evidence']}", at)
        set_status(production, "shadow", f"demoted by {shadow['model_version']}: {decision['evidence']}", at)
    else:
        set_status(shadow, "production", f"rolled back: {decision['evidence']}", at)
        set_status(production, "shadow", f"rolled back from production: {decision['evidence']}", at)
    return registry


def decide_actions(start, end, config):
    """Thresholds and actions on the labelled day; writes the actions table.
    apply_actions carries out promote and rollback; the DAG triggers model_training for a retrain."""
    first, last = labelled_day(start, end, config)
    mon = config["monitoring"]
    day = [("date", "==", str(first.date()))]
    actions = []

    def act(action, model_version, reason, evidence):
        actions.append({"at": utc_now(), "action": action, "model_version": model_version,
                        "reason": reason, "evidence": evidence})

    # 1. data quality: any alert in the day's hourly runs
    dq = read_table("monitoring/data_quality", config, filters=day)
    for r in dq[dq["status"] == "alert"].itertuples():
        act("alert", None, f"data quality {r.check} failed in {r.run_id}", f"{r.check}={r.value:g}")
    # 2. stability over the last week: the forecasts' PSI, and the features' CSI, above the alert band
    stability = read_table("monitoring/stability", config, filters=day)
    if not stability.empty:  # nothing in the first week, before a model has a week of forecasts
        alerts = stability[stability["status"] == "alert"]
        for r in alerts[alerts["index"] == "PSI"].itertuples():
            act("alert", r.model_version, "forecast distribution shifted (PSI)", f"PSI={r.value:.3f}")
        # CSI alerts only for the model's key features: its largest mean |SHAP| in training. Seasonal
        # inputs (a winter week's temperatures) shift every week without changing the forecast
        for model_version, rows in alerts[alerts["index"] == "CSI"].groupby("model_version"):
            entry = next(e for e in read_registry(config) if e["model_version"] == model_version)
            baseline = load_artefact(entry)["shap_baseline"]
            key = sorted(baseline, key=baseline.get, reverse=True)[:mon["csi_key_features"]]
            rows = rows[rows["variable"].isin(key)]
            if not rows.empty:
                act("alert", model_version, "key feature distributions shifted (CSI): " + ", ".join(rows["variable"]),
                    f"CSI max={rows['value'].max():.3f}")
    # 3. attribution: nDCG below the threshold on ndcg_alert_days days in a row
    attribution = read_table("monitoring/attribution", config,
                             filters=[("date", ">", str((first - timedelta(days=mon["ndcg_alert_days"])).date())),
                                      ("date", "<=", str(first.date()))])
    ndcg_rows = attribution[attribution["feature"] == "__ndcg__"]
    for model_version, rows in ndcg_rows.groupby("model_version"):
        if len(rows) >= mon["ndcg_alert_days"] and (rows["status"] == "alert").all():
            act("alert", model_version, "feature attribution drifted (nDCG)", f"nDCG={rows['value'].iloc[-1]:.3f}")
    # the rolling P0 over the promotion window, per model and role, from today's performance rows
    performance = read_table("monitoring/model_performance", config, filters=day)
    rolling = performance[(performance["granularity"] == f"rolling_{mon['p0_window_days']}d")
                          & (performance["metric"] == "P0") & (performance["segment"] == "citywide")]
    rolling_p0 = {(r.model_version, r.role): r.value for r in rolling.itertuples()}
    registry = read_registry(config)
    production = next((e for e in registry if e["status"] == "production"), None)
    # 4. performance: production's rolling P0 against its validation P0 -> retrain, at most once per cooldown
    if production is not None and (production["model_version"], "production") in rolling_p0:
        p0 = rolling_p0[(production["model_version"], "production")]
        validation_p0 = load_artefact(production)["metrics"]["validation"]["P0"]
        drop = (validation_p0 - p0) / validation_p0 * 100
        recent = read_table("monitoring/actions", config,
                            filters=[("date", ">", str((first - timedelta(days=mon["retrain_cooldown_days"])).date())),
                                     ("date", "<", str(first.date()))])
        retrained = "action" in recent and (recent["action"] == "retrain").any()
        if drop > mon["p0_drop_pct"] and not retrained:
            act("retrain", production["model_version"], f"rolling {mon['p0_window_days']}-day P0 "
                f"{drop:.1f}% below validation", f"P0={p0:.3f}")
        print(f"decide_actions {first:%Y-%m-%d}: production rolling P0 {p0:.3f} (validation P0 {validation_p0:.3f})")
    # 5. promote or roll back
    decision = swap_decision(registry, rolling_p0, config)
    if decision is not None:
        act(decision["action"], decision["model_version"], decision["reason"], decision["evidence"])
    # 6. bias drift: the P0 gap between the high-taxi-coverage areas and the rest, against the same
    # gap on validation. The busiest areas always fill the top k, so the gap itself is expected;
    # an alert means it widened
    if production is not None:
        # the current production model only: on the day after a swap, the old one forecast an hour too
        fairness = performance[(performance["granularity"] == "day") & (performance["metric"] == "P0")
                               & (performance["model_version"] == production["model_version"])
                               & (performance["role"] == "production")]
        groups = fairness.set_index("segment")["value"]
    if production is not None and {"high_taxi_coverage", "other_areas"} <= set(groups.index):
        reference = load_artefact(production)["metrics"]["validation"]["groups"]
        gap = groups["high_taxi_coverage"] - groups["other_areas"]
        reference_gap = reference["high_taxi_coverage"]["P0"] - reference["other_areas"]["P0"]
        if gap - reference_gap > mon["fairness_gap"]:
            act("alert", production["model_version"], "fairness gap between area groups widened",
                f"P0 gap={gap:.3f} (validation {reference_gap:.3f})")

    # explicit types: a day without actions must still write the columns' types, or reading the table
    # across days fails (an empty column has no type)
    df = pd.DataFrame(actions, columns=["at", "action", "model_version", "reason", "evidence"]).astype(
        {"at": "datetime64[us]", "action": "string", "model_version": "string", "reason": "string", "evidence": "string"})
    write_rows(df, "monitoring/actions", first.date(), run_id_for(start, end), "decide_actions", config)
    print(f"decide_actions {first:%Y-%m-%d}: {len(df)} action(s)")
    for a in actions:
        print(f"  {a['action']}: {a['reason']} ({a['evidence']})")
    return len(df)


def apply_actions(start, end, config):
    """Carries out the labelled day's promote or rollback in the registry. The next hourly run serves
    with the swapped models. A rerun changes nothing: the registry no longer matches the decision."""
    first, _ = labelled_day(start, end, config)
    actions = read_table("monitoring/actions", config, filters=[("date", "==", str(first.date()))])
    swaps = actions[actions["action"].isin(["promote", "rollback"])]
    registry = read_registry(config)
    shadow = next((e for e in registry if e["status"] == "shadow"), None)
    for a in swaps.itertuples():
        if shadow is None or shadow["model_version"] != a.model_version:
            print(f"apply_actions {first:%Y-%m-%d}: {a.action} {a.model_version} already applied")
            continue
        apply_swap(registry, {"action": a.action, "evidence": a.evidence}, utc_now().isoformat())
        write_registry(registry, config)
        print(f"apply_actions {first:%Y-%m-%d}: {a.action}: {a.model_version} -> production ({a.evidence})")
    if swaps.empty:
        print(f"apply_actions {first:%Y-%m-%d}: no promotion or rollback")
    return len(swaps)
