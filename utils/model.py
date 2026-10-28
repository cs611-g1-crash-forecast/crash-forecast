import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.dataset
import xgboost
from dateutil.relativedelta import relativedelta
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import StandardScaler

LABEL = "crash_count"


# reading the gold stores (pandas: the data for training fits in memory)

def read_gold(table, columns, first, end, config):
    """Rows of a gold table with hour_start in [first, end), only the columns asked for."""
    path = os.path.join(config["paths"]["datamart"], "gold", table)
    # the date partitions read back as text: compare ISO dates as text
    filters = [("date", ">=", str(first.date())), ("date", "<=", str(end.date()))]
    df = pd.read_parquet(path, columns=columns, filters=filters)
    return df[(df["hour_start"] >= first) & (df["hour_start"] < end)]


def read_labels(first, end, config):
    """The label store, taking each hour's latest snapshot."""
    df = read_gold("label_store", ["community_area", "hour_start", LABEL, "label_snapshot_date"], first, end, config)
    df = df.sort_values("label_snapshot_date").drop_duplicates(["community_area", "hour_start"], keep="last")
    return df.drop(columns="label_snapshot_date")


def feature_columns(config):
    """Every feature of the families switched on in config (features are selected by family)."""
    path = os.path.join(config["paths"]["datamart"], "gold", "feature_store")
    names = pyarrow.dataset.dataset(path, partitioning="hive").schema.names
    return sorted(c for c in names if c.split("_")[0] in config["features"]["families"])


def split_bounds(split, config, shift=0):
    """[first, end) of a split given as inclusive months, e.g. ["2018-04", "2021-12"],
    moved forward by shift months (a retrain after go-live)."""
    first, last = config["splits"][split]
    return (datetime.strptime(first, "%Y-%m") + relativedelta(months=shift),
            datetime.strptime(last, "%Y-%m") + relativedelta(months=shift + 1))


def load_split(split, features, config, shift=0):
    first, end = split_bounds(split, config, shift)
    X = read_gold("feature_store", ["community_area", "hour_start"] + features, first, end, config)
    y = read_labels(first, end, config)
    return X.merge(y, on=["community_area", "hour_start"], how="inner")


# the models

class SeasonalBaseline:
    """B0: an area's mean count for the same hour of the week over the previous 4 weeks.
    That is the feature crash_area_how_mean_4w, so B0 needs no fitting."""
    feature = "crash_area_how_mean_4w"

    def predict(self, X):
        return X[self.feature].fillna(0).to_numpy()


def fit_candidate(algorithm, train, features, config):
    """Returns (model, feature_list, preprocessing_transformers), all fitted on train only."""
    if algorithm == "B0":
        return SeasonalBaseline(), [SeasonalBaseline.feature], {}
    if algorithm == "M1":
        # Poisson regression on standardised features; missing values take the train median.
        # untuned: sklearn's default regularisation
        imputer = SimpleImputer(strategy="median").fit(train[features])
        scaler = StandardScaler().fit(imputer.transform(train[features]))
        model = PoissonRegressor(alpha=1e-4, max_iter=300)
        model.fit(scaler.transform(imputer.transform(train[features])), train[LABEL])
        return model, features, {"imputer": imputer, "stdscaler": scaler}
    if algorithm == "M2":
        # gradient-boosted trees with a Poisson objective; XGBoost handles missing values itself,
        # and trees need no scaling, so there is no preprocessing
        model = xgboost.XGBRegressor(objective="count:poisson", tree_method="hist", random_state=0,
                                     **config["models"]["params"]["M2"])
        model.fit(train[features], train[LABEL])
        return model, features, {}
    raise ValueError(f"unknown algorithm {algorithm}")


def predict(artefact, X):
    """Expected crash counts for the rows of X, with the artefact's own preprocessing."""
    X = X[artefact["feature_list"]]
    pre = artefact["preprocessing_transformers"]
    if pre:
        X = pre["stdscaler"].transform(pre["imputer"].transform(X))
    return artefact["model"].predict(X)


def shap_values(artefact, X):
    """Per-row contributions of each feature to the prediction.
    Linear models have exact SHAP values: coef x (x - training mean), on the model's (log) scale.
    B0's prediction is its one feature, so its contribution is x - mean."""
    features = artefact["feature_list"]
    pre = artefact["preprocessing_transformers"]
    if isinstance(artefact["model"], xgboost.XGBRegressor):
        # TreeSHAP, built into XGBoost; the last column is the bias term, not a feature
        contribs = artefact["model"].get_booster().predict(xgboost.DMatrix(X[features]), pred_contribs=True)
        return pd.DataFrame(contribs[:, :-1], columns=features)
    if not pre:
        x = X[features].fillna(0).to_numpy()
        return pd.DataFrame(x - artefact["train_means"], columns=features)
    z = pre["stdscaler"].transform(pre["imputer"].transform(X[features]))  # scaled: train mean is 0
    return pd.DataFrame(z * artefact["model"].coef_, columns=features)


# metrics

def top_k_capture(df, config):
    """P0: per 8-hour shift, the share of the shift's crashes in the k areas with the highest forecast.
    Returns the mean over shifts that had at least one crash, and the per-row flag 'in the top k'."""
    k, shift_hours = config["evaluation"]["top_k"], config["evaluation"]["shift_hours"]
    df = df.assign(shift=df["hour_start"].dt.floor("D") + pd.to_timedelta(df["hour_start"].dt.hour // shift_hours * shift_hours, unit="h"))
    shift = df.groupby(["shift", "community_area"], as_index=False)[["actual", "predicted"]].sum()
    # rank the areas by forecast within each shift (ties: lower area number first)
    shift = shift.sort_values(["shift", "predicted", "community_area"], ascending=[True, False, True])
    shift["in_top_k"] = shift.groupby("shift").cumcount() < k
    per_shift = shift.groupby("shift").apply(
        lambda s: s.loc[s["in_top_k"], "actual"].sum() / s["actual"].sum() if s["actual"].sum() > 0 else np.nan,
        include_groups=False)
    return per_shift.mean(), shift


def metrics(df, config, groups):
    """P0, MAE, RMSE, Poisson deviance and bias for actual vs predicted; P0 and MAE per fairness group.
    A group's P0 is the share of its crashes that fell in the citywide top k (is it covered as well as the rest?)."""
    p0, shift = top_k_capture(df, config)
    err = df["predicted"] - df["actual"]
    result = {
        "P0": p0,
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        # deviance needs a positive forecast: an hour forecast at 0 is scored as 0.001
        "poisson_deviance": mean_poisson_deviance(df["actual"], df["predicted"].clip(lower=1e-3)),
        "bias": df["predicted"].sum() / df["actual"].sum(),
        "groups": {},
    }
    for name, areas in groups.items():
        g = shift[shift["community_area"].isin(areas)]
        rows = df[df["community_area"].isin(areas)]
        result["groups"][name] = {
            "P0": g.loc[g["in_top_k"], "actual"].sum() / g["actual"].sum(),
            "MAE": (rows["predicted"] - rows["actual"]).abs().mean(),
        }
    return result


def psi_reference(values, bins=10):
    """The training distribution of a variable for PSI: inner bin edges at its deciles, and each bin's share."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))[1:-1])
    shares = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1) / len(values)
    return {"edges": edges.tolist(), "shares": shares.tolist()}


def psi(values, reference):
    """Population stability index of values against a psi_reference (L7: < 0.1 stable, > 0.25 shifted)."""
    edges = np.array(reference["edges"])
    expected = np.clip(np.array(reference["shares"]), 1e-4, None)
    actual = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1) / len(values)
    actual = np.clip(actual, 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def fairness_groups(train, config):
    """The fairness groups: high taxi coverage vs the rest; crash-rate terciles from train."""
    high_taxi = config["evaluation"]["fairness_groups"]["high_taxi_coverage"]
    groups = {"high_taxi_coverage": high_taxi,
              "other_areas": [a for a in range(1, 78) if a not in high_taxi]}
    rate = train.groupby("community_area")[LABEL].mean()
    terciles = pd.qcut(rate, 3, labels=["low", "mid", "high"])
    for t in ["low", "mid", "high"]:
        groups[f"crash_rate_{t}"] = sorted(terciles[terciles == t].index.tolist())
    return groups


# the registry

def registry_path(config):
    return os.path.join(config["paths"]["model_bank"], "registry.json")


def read_registry(config):
    if not os.path.exists(registry_path(config)):
        return []
    with open(registry_path(config)) as f:
        return json.load(f)


def write_registry(registry, config):
    os.makedirs(config["paths"]["model_bank"], exist_ok=True)
    with open(registry_path(config), "w") as f:
        json.dump(registry, f, indent=2, default=str)


def set_status(entry, status, reason, at):
    entry["status"] = status
    entry["status_history"].append({"status": status, "at": at, "reason": reason})


def model_with_status(status, config):
    """The registry entry with this status (production or shadow), or None."""
    return next((e for e in read_registry(config) if e["status"] == status), None)


def load_artefact(entry):
    with open(entry["artefact_path"], "rb") as f:
        return pickle.load(f)


def utc_now():
    # audit times are UTC, naive
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


# ---------- choosing a model ----------

def select_candidate(entries, config):
    """The candidate to put in shadow: the best validation P0 among those that beat B0's by the margin.
    P0s within the tie of the best are decided by the lower Poisson deviance. None if none beats B0.
    Validation, not test: test is kept for the final comparison, so choosing on it would flatter it."""
    rule = config["models"]["selection"]
    b0 = next(e for e in entries if e["algorithm"] == "B0")
    bar = b0["validation_metrics"]["P0"] + rule["margin"]
    eligible = [e for e in entries if e is not b0 and e["validation_metrics"]["P0"] >= bar]
    if not eligible:
        return None
    best = max(e["validation_metrics"]["P0"] for e in eligible)
    close = [e for e in eligible if best - e["validation_metrics"]["P0"] <= rule["tie"]]
    return min(close, key=lambda e: e["validation_metrics"]["poisson_deviance"])


# the training job

def train_models(start, end, config):
    """Train every candidate on the config splits (all before end), evaluate, and register them.
    The selection rule picks the candidate that shadows production; the very first run also puts B0
    in production. Promoting the shadow is the monitoring's decision, not training's."""
    # a retrain after go-live moves every split forward by the whole months since go-live:
    # the newest labelled months become test, and the oldest leave train
    go_live = datetime.fromisoformat(config["deployment"]["go_live"])
    shift = max(0, (end.year - go_live.year) * 12 + end.month - go_live.month)
    for split in ["train", "validation", "test"]:
        assert split_bounds(split, config, shift)[1] <= end, f"{split} ends after this run's end {end}"
    run_at = utc_now()
    features = feature_columns(config)
    data = {split: load_split(split, features, config, shift) for split in ["train", "validation", "test"]}
    
    train_rows = data["train"]
    if config["splits"]["train_exclude"]:
        first, last = config["splits"]["train_exclude"]
        month = train_rows["hour_start"].dt.strftime("%Y-%m")
        train_rows = train_rows[~month.between(first, last)]
    groups = fairness_groups(train_rows, config)
    print(f"train: {len(train_rows):,} rows, validation: {len(data['validation']):,}, test: {len(data['test']):,}; "
          f"{len(features)} features")

    registry = read_registry(config)
    train_end = split_bounds("train", config, shift)[1] - relativedelta(months=1)
    entries = []
    for algorithm in config["models"]["candidates"]:
        model, feature_list, pre = fit_candidate(algorithm, train_rows, features, config)
        model_version = f"{algorithm}_{train_end:%Y_%m}_{run_at:%Y%m%dT%H%M%S}"
        artefact = {
            "model": model,
            "model_version": model_version,
            "preprocessing_transformers": pre,
            "feature_list": feature_list,
            "config_snapshot": config,
            "data_dates": {s: [str(b) for b in split_bounds(s, config, shift)]
                           for s in ["train", "validation", "test"]},
            "train_means": train_rows[feature_list].fillna(0).mean().to_numpy(),
        }
        artefact["metrics"] = {}
        for split, df in data.items():
            scored = pd.DataFrame({"community_area": df["community_area"], "hour_start": df["hour_start"],
                                   "actual": df[LABEL], "predicted": predict(artefact, df)})
            artefact["metrics"][split] = metrics(scored, config, groups)
        # the SHAP baseline: mean |SHAP| per feature on the training data (for the attribution monitor)
        artefact["shap_baseline"] = shap_values(artefact, train_rows).abs().mean().to_dict()
        # what monitoring compares against: the training forecasts' distribution (PSI), each feature's
        # (CSI; only the values it had, as a missing value has no bin), and the area groups
        artefact["stability_reference"] = {
            "prediction": psi_reference(predict(artefact, train_rows)),
            "features": {f: psi_reference(train_rows[f].dropna().to_numpy()) for f in feature_list},
        }
        artefact["fairness_groups"] = groups

        # save the model artefact - IRL to a model registry (S3 / MLflow)
        model_dir = os.path.join(config["paths"]["model_bank"], model_version)
        os.makedirs(model_dir, exist_ok=True)
        artefact_path = os.path.join(model_dir, "model_artefact.pkl")
        with open(artefact_path, "wb") as f:
            pickle.dump(artefact, f)

        entry = {"model_version": model_version, "algorithm": algorithm, "artefact_path": artefact_path,
                 "status": None, "created_at": run_at.isoformat(), "status_history": [],
                 "validation_metrics": artefact["metrics"]["validation"],
                 "test_metrics": artefact["metrics"]["test"]}
        set_status(entry, "candidate", "trained", run_at.isoformat())
        entries.append(entry)
        m = artefact["metrics"]
        print(f"{model_version}: P0 validation {m['validation']['P0']:.3f}, test {m['test']['P0']:.3f}; "
              f"MAE test {m['test']['MAE']:.4f}; bias test {m['test']['bias']:.3f} -> {artefact_path}")

    now = run_at.isoformat()
    b0 = next(e for e in entries if e["algorithm"] == "B0")
    winner = select_candidate(entries, config)
    if not any(e["status"] == "production" for e in registry):
        # first deployment: the baseline serves first
        set_status(b0, "production", "first deployment: the baseline serves first", now)
    else:
        set_status(b0, "rejected", "the baseline: it sets the bar for the others", now)
    if winner is not None:
        # the selected candidate shadows production; promotion is the monitoring's decision
        for e in registry:
            if e["status"] == "shadow":
                set_status(e, "archived", f"replaced in shadow by {winner['model_version']}", now)
        set_status(winner, "shadow", f"selected: validation P0 {winner['validation_metrics']['P0']:.3f}, "
                   f"B0 {b0['validation_metrics']['P0']:.3f}", now)
    for e in entries:
        if e["status"] == "candidate":
            set_status(e, "rejected", "not selected: " + (f"{winner['model_version']} was" if winner else
                       "no candidate beat B0 by the margin, so none was") + " selected", now)
    registry.extend(entries)
    write_registry(registry, config)
    for e in entries:
        print(f"registry: {e['model_version']} -> {e['status']}")
    return entries
