from datetime import timedelta

import numpy as np
import pandas as pd

from utils.bronze import run_id_for
from utils.gold import label_hours
from utils.model import (LABEL, load_artefact, metrics, psi, read_gold, read_labels, read_registry,
                         shap_values, utc_now)
from utils.tables import read_table, write_rows


def labelled_day(start, end, config):
    """The daily run for [D, D+1) monitors day D-1: the day gold_labels has just labelled."""
    first, end_hour, _ = label_hours(start, end, config)
    return first, end_hour


def scored_rows(start, end, config):
    """Every forecast of the labelled day (both roles) with its label."""
    first, last = labelled_day(start, end, config)
    predictions = read_table("model_predictions", config,
                             filters=[("date", ">=", str(first.date())), ("date", "<", str(last.date()))])
    predictions = predictions[(predictions["hour_start"] >= first) & (predictions["hour_start"] < last)]
    labels = read_labels(first, last, config)
    snapshot = read_gold("label_store", ["hour_start", "label_snapshot_date"], first, last, config)["label_snapshot_date"].max()
    rows = predictions.merge(labels, on=["community_area", "hour_start"], how="inner")
    return rows.rename(columns={LABEL: "actual", "predicted_count": "predicted"}), snapshot


def period(start, end, config):
    first, last = labelled_day(start, end, config)
    return {"period_start": first, "period_end": last, "granularity": "day"}


def models_scored(rows):
    """The (model_version, role) pairs that forecast the labelled day."""
    return rows[["model_version", "role"]].drop_duplicates().itertuples(index=False)


def monitor_performance(start, end, config):
    """Model performance: P0 and the error metrics (MAE, RMSE, Poisson deviance, bias) on the labelled day, per model and role."""
    rows, snapshot = scored_rows(start, end, config)
    out = []
    for model_version, role in models_scored(rows):
        m = metrics(rows[rows["model_version"] == model_version], config, {})
        for metric in ["P0", "MAE", "RMSE", "poisson_deviance", "bias"]:
            out.append({**period(start, end, config), "model_version": model_version, "role": role,
                        "segment": "citywide", "metric": metric, "value": float(m[metric]),
                        "label_snapshot_date": snapshot})
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


def monitor_stability(start, end, config):
    """Stability: PSI of the forecasts on the labelled day against the training window."""
    rows, _ = scored_rows(start, end, config)
    thresholds = config["monitoring"]["psi"]
    out = []
    for model_version, role in models_scored(rows):
        entry = next(e for e in read_registry(config) if e["model_version"] == model_version)
        reference = load_artefact(entry)["stability_reference"]["prediction"]
        value = psi(rows.loc[rows["model_version"] == model_version, "predicted"], reference)
        status = "alert" if value > thresholds["alert"] else "warn" if value > thresholds["warn"] else "ok"
        out.append({**period(start, end, config), "model_version": model_version, "variable": "prediction",
                    "index": "PSI", "value": value, "status": status})
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
    df = pd.DataFrame(out)
    first, _ = labelled_day(start, end, config)
    write_rows(df, f"monitoring/{table}", first.date(), run_id_for(start, end), task, config)
    for r in out:
        name = r.get("metric") or r.get("index") or r.get("feature")
        parts = [r["model_version"], r.get("role"), r.get("segment"), f"{name} = {r['value']:.4f}", r.get("status")]
        print(f"{task} {first:%Y-%m-%d}: " + " ".join(p for p in parts if p))
    return len(out)


def decide_actions(start, end, config):
    """Thresholds and actions, one check of each kind; writes the actions table.
    The follow-through (triggering model_training, changing the registry) is not built yet."""
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
    # 2. stability: PSI above the alert band
    stability = read_table("monitoring/stability", config, filters=day)
    for r in stability[stability["status"] == "alert"].itertuples():
        act("alert", r.model_version, "forecast distribution shifted (PSI)", f"PSI={r.value:.3f}")
    # 3. attribution: nDCG below the threshold on ndcg_alert_days days in a row
    attribution = read_table("monitoring/attribution", config,
                             filters=[("date", ">", str((first - timedelta(days=mon["ndcg_alert_days"])).date())),
                                      ("date", "<=", str(first.date()))])
    ndcg_rows = attribution[attribution["feature"] == "__ndcg__"]
    for model_version, rows in ndcg_rows.groupby("model_version"):
        if len(rows) >= mon["ndcg_alert_days"] and (rows["status"] == "alert").all():
            act("alert", model_version, "feature attribution drifted (nDCG)", f"nDCG={rows['value'].iloc[-1]:.3f}")
    # 4. performance: rolling P0 of the production model against its validation P0 -> retrain
    performance = read_table("monitoring/model_performance", config,
                             filters=[("date", ">", str((first - timedelta(days=mon["p0_window_days"])).date())),
                                      ("date", "<=", str(first.date()))])
    p0 = performance[(performance["metric"] == "P0") & (performance["segment"] == "citywide")
                     & (performance["role"] == "production")]
    production = next((e for e in read_registry(config) if e["status"] == "production"), None)
    if production is not None:
        daily = p0.loc[p0["model_version"] == production["model_version"], "value"]
        validation_p0 = load_artefact(production)["metrics"]["validation"]["P0"]
        if len(daily) >= mon["p0_window_days"]:
            drop = (validation_p0 - daily.mean()) / validation_p0 * 100
            if drop > mon["p0_drop_pct"]:
                act("retrain", production["model_version"], f"rolling {mon['p0_window_days']}-day P0 "
                    f"{drop:.1f}% below validation", f"P0={daily.mean():.3f}")
        print(f"decide_actions {first:%Y-%m-%d}: production rolling P0 over {len(daily)} of "
              f"{mon['p0_window_days']} days (validation P0 {validation_p0:.3f})")
    # 5. bias drift: the P0 gap between the high-taxi-coverage areas and the rest, against the same
    # gap on validation. The busiest areas always fill the top k, so the gap itself is expected;
    # an alert means it widened
    fairness = performance[(performance["date"].astype(str) == str(first.date()))
                           & (performance["metric"] == "P0") & (performance["role"] == "production")]
    groups = fairness.set_index("segment")["value"]
    if production is not None and {"high_taxi_coverage", "other_areas"} <= set(groups.index):
        reference = load_artefact(production)["metrics"]["validation"]["groups"]
        gap = groups["high_taxi_coverage"] - groups["other_areas"]
        reference_gap = reference["high_taxi_coverage"]["P0"] - reference["other_areas"]["P0"]
        if gap - reference_gap > mon["fairness_gap"]:
            act("alert", production["model_version"], "fairness gap between area groups widened",
                f"P0 gap={gap:.3f} (validation {reference_gap:.3f})")

    df = pd.DataFrame(actions, columns=["at", "action", "model_version", "reason", "evidence"])
    write_rows(df, "monitoring/actions", first.date(), run_id_for(start, end), "decide_actions", config)
    print(f"decide_actions {first:%Y-%m-%d}: {len(df)} action(s)")
    for a in actions:
        print(f"  {a['action']}: {a['reason']} ({a['evidence']})")
    return len(df)
