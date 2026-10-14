"""M4 - train, compare and deploy the rating-decline forecasters.

Models compared on an out-of-time test period (no temporal leakage: the
split is by calendar month and every feature only uses past data):

    majority            constant base rate                                   (floor)
    current_negativity  share of recent reviews with any complaint, no model (naive)
    rating_logreg       logistic regression, rating-trend features only     (baseline)
    rating_lgbm         LightGBM, rating-trend features only                  (baseline)
    sentiment_lgbm      LightGBM, rating + aspect-agnostic negativity         (baseline)
    radview_lgbm        LightGBM, all features incl. per-aspect M2/M3 signals
    radview_gru         PyTorch GRU ensemble over 12-month sequences (Radview)

Every model is Platt-calibrated on the validation months so probabilities are
comparable. Metrics: AUROC, PR-AUC (the honest metric for a rare outcome),
Brier score (calibration), precision/recall in the top 10% riskiest
business-months, early warning at a fixed 15% alert budget (share of decline
onsets flagged up to 6 months ahead, lead time) and - on synthetic data, where
the injected deterioration episodes are known - how many episodes were caught
*before* the drop became visible in the 3-month rating.

Artefacts:
    models/m4_gru.pt, models/m4_lgbm.txt
    outputs/m4_evaluation.json, outputs/m4_test_predictions.parquet
    outputs/risk_history.parquet   risk score for every business-month
    outputs/risk_latest.parquet    latest risk + SHAP + counterfactual + star cost per aspect

Usage:  python -m radview.m4_forecast.train
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from radview.config import (ASPECTS, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, RANDOM_SEED, RAW_DIR, RISK_HIGH_LIFT,
                            RISK_MEDIUM_LIFT, TEST_MONTHS, VALID_MONTHS)
from radview.m4_forecast.attribution import (counterfactual_attribution, fit_star_impact, method_agreement,
                                             star_costs)
from radview.m4_forecast.features import ALL_FEATURES, RATING_FEATURES, SENTIMENT_FEATURES
from radview.m4_forecast.model import (DEVICE, DeclineGRU, GRUEnsemble, SequenceData, apply_platt, explain,
                                       fit_platt, predict, train_gru)

N_SEEDS = 3             # GRU ensemble size
ALERT_BUDGET = 0.15     # share of business-months each model may flag in the early-warning tests
LOOKBACK_ALERT = 6      # months before an onset in which an alert counts
VISIBLE_DROP = 0.3      # stars: when an injected episode "becomes visible" in the 3-month rating

FEATURE_GROUPS = {
    **{a: [f for f in ALL_FEATURES if f.startswith(a + "_")] for a in ASPECTS},
    "rating trend": [f for f in RATING_FEATURES if f.startswith("rating_")],
    "volume & footfall": ["n_reviews_3m", "review_velocity_ratio", "months_active", "checkins_3m", "checkin_ratio", "tips_3m"],
    "overall sentiment": ["low_star_share_3m", "any_neg_share_3m", "any_neg_share_change", "n_spiking_aspects"],
}


# --------------------------------------------------------------------------
# Split and baselines
# --------------------------------------------------------------------------
def time_split(feat: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    months = sorted(feat.loc[feat["decline"].notna(), "month"].unique())
    test_start = months[-TEST_MONTHS]
    valid_start = months[-TEST_MONTHS - VALID_MONTHS]
    labelled = feat["decline"].notna()
    train = labelled & (feat["month"] < valid_start)
    valid = labelled & (feat["month"] >= valid_start) & (feat["month"] < test_start)
    test = labelled & (feat["month"] >= test_start)
    info = {"train": f"{months[0]}..{months[months.index(valid_start) - 1]}",
            "valid": f"{valid_start}..{months[months.index(test_start) - 1]}",
            "test": f"{test_start}..{months[-1]}", "test_start": test_start,
            "n_train": int(train.sum()), "n_valid": int(valid.sum()), "n_test": int(test.sum()),
            "pos_rate_train": round(float(feat.loc[train, "decline"].mean()), 4),
            "pos_rate_test": round(float(feat.loc[test, "decline"].mean()), 4)}
    return train, valid, test, info


def _lgbm(X_tr, y_tr, X_va, y_va):
    pos = y_tr.mean()
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             scale_pos_weight=(1 - pos) / pos, random_state=RANDOM_SEED, verbose=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def early_warning(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict:
    """Share of decline onsets with an alert in the preceding LOOKBACK_ALERT months, and the lead time."""
    df = frame[["business_id", "month", "decline"]].copy()
    df["alert"] = prob >= threshold
    df = df.sort_values(["business_id", "month"])
    leads = []
    for _, g in df.groupby("business_id", sort=False):
        d = g["decline"].to_numpy()
        a = g["alert"].to_numpy()
        for i in range(1, len(g)):
            if d[i] == 1 and d[i - 1] == 0:  # onset: the decline window starts after month i
                lo = max(0, i - LOOKBACK_ALERT + 1)
                hits = np.flatnonzero(a[lo:i + 1])
                leads.append((i + 1) - (lo + hits[0]) if len(hits) else np.nan)
    leads = np.array(leads, dtype=float)
    det = ~np.isnan(leads)
    return {"onsets": int(len(leads)), "detected_pct": round(100 * float(det.mean()), 1) if len(leads) else None,
            "median_lead_months": float(np.median(leads[det])) if det.any() else None,
            "mean_lead_months": round(float(np.mean(leads[det])), 2) if det.any() else None}


def metrics(y: np.ndarray, p: np.ndarray, frame: pd.DataFrame) -> dict:
    k = max(1, int(0.10 * len(p)))
    top = np.argsort(-p, kind="stable")[:k]
    thr = np.quantile(p, 1 - ALERT_BUDGET)
    single = len(np.unique(p)) == 1
    return {"auroc": 0.5 if single else round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "precision_at_10pct": round(float(y[top].mean()), 4),
            "recall_at_10pct": round(float(y[top].sum() / max(y.sum(), 1)), 4),
            "early_warning": None if single else early_warning(frame, p, thr)}


def episode_evaluation(feat: pd.DataFrame, probs: dict[str, np.ndarray], test_start: str) -> dict | None:
    """Synthetic data only: were the injected deterioration episodes flagged before they showed in the rating?

    For each episode starting in the test period we find the month its damage became *visible*
    (3-month rating >= VISIBLE_DROP below the 6-month rating just before onset) and check, per
    model, when the first alert fired (alert = score in the model's top ALERT_BUDGET of test months).
    """
    path = RAW_DIR / "_ground_truth_episodes.json"
    if not path.exists():
        return None
    episodes = [e for e in json.loads(path.read_text()) if e["sign"] < 0 and e["onset_month"] >= test_start]
    idx = {(b, m): i for i, (b, m) in enumerate(zip(feat["business_id"], feat["month"]))}
    test_rows = (feat["month"] >= test_start).to_numpy()
    out = {}
    for name, p in probs.items():
        if np.unique(p[test_rows]).size == 1:
            continue  # a constant score cannot rank months, so "alerts" would be meaningless
        thr = np.quantile(p[test_rows], 1 - ALERT_BUDGET)
        n = caught = before = 0
        leads = []
        for e in episodes:
            start = idx.get((e["business_id"], e["onset_month"]))
            if start is None or start == 0 or feat["business_id"].iat[start - 1] != e["business_id"]:
                continue
            base = feat["rating_6m"].iat[start - 1]
            visible = None
            for j in range(start, min(start + 12, len(feat))):
                if feat["business_id"].iat[j] != e["business_id"]:
                    break
                if feat["rating_3m"].iat[j] <= base - VISIBLE_DROP:
                    visible = j
                    break
            if visible is None:
                continue  # never visibly hurt the rating - nothing to warn about
            n += 1
            alerts = [j for j in range(start, visible + 3) if j < len(feat)
                      and feat["business_id"].iat[j] == e["business_id"] and p[j] >= thr]
            if alerts:
                caught += 1
                leads.append(visible - alerts[0])
                before += alerts[0] < visible
        out[name] = {"episodes": n, "detected_pct": round(100 * caught / n, 1) if n else None,
                     "before_visible_pct": round(100 * before / n, 1) if n else None,
                     "median_lead_months": float(np.median(leads)) if leads else None}
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def run(epochs: int = 30) -> dict:
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet").sort_values(["business_id", "month"]).reset_index(drop=True)
    tr, va, te, split = time_split(feat)
    print("[M4] split:", split)
    y = feat["decline"].fillna(0).to_numpy().astype(int)
    rows_tr, rows_va, rows_te = (np.flatnonzero(m.to_numpy()) for m in (tr, va, te))
    all_rows = np.arange(len(feat))
    raw_all: dict[str, np.ndarray] = {}   # uncalibrated scores for every business-month

    # ---------------- baselines ----------------------------------------------
    raw_all["majority"] = np.full(len(feat), y[rows_tr].mean())
    raw_all["current_negativity"] = feat["any_neg_share_3m"].fillna(0).to_numpy()
    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
    logreg.fit(feat.loc[tr, RATING_FEATURES], y[tr])
    raw_all["rating_logreg"] = logreg.predict_proba(feat[RATING_FEATURES])[:, 1]
    lgbm_models = {}
    for name, cols in (("rating_lgbm", RATING_FEATURES), ("sentiment_lgbm", SENTIMENT_FEATURES),
                       ("radview_lgbm", ALL_FEATURES)):
        m = _lgbm(feat.loc[tr, cols], y[tr], feat.loc[va, cols], y[va])
        lgbm_models[name] = m
        raw_all[name] = m.predict_proba(feat[cols])[:, 1]

    # ---------------- Radview GRU (PyTorch) ---------------------------------
    mean = feat.loc[tr, ALL_FEATURES].mean()
    std = feat.loc[tr, ALL_FEATURES].std().replace(0, 1).fillna(1)
    data = SequenceData(feat, ALL_FEATURES, mean, std)
    members, train_info = [], []
    for k in range(N_SEEDS):
        m, info = train_gru(data, y, rows_tr, rows_va, epochs=epochs, seed=RANDOM_SEED + k)
        members.append(m)
        train_info.append(info)
    gru = GRUEnsemble(members).eval()
    raw_all["radview_gru"] = predict(gru, data, all_rows)

    # ---------------- calibrate on validation, evaluate on test ---------------
    calibs, probs, results = {}, {}, {}
    for name, raw in raw_all.items():
        calibs[name] = None if name == "majority" else fit_platt(raw[rows_va], y[rows_va])
        probs[name] = raw if calibs[name] is None else apply_platt(raw, calibs[name])
    test_frame = feat.loc[te].reset_index(drop=True)
    for name, p in probs.items():
        results[name] = r = metrics(y[rows_te], p[rows_te], test_frame)
        ew = r["early_warning"] or {}
        print(f"[M4] {name:18s} AUROC {r['auroc']:.3f}  PR-AUC {r['pr_auc']:.3f}  Brier {r['brier']:.4f}  "
              f"P@10% {r['precision_at_10pct']:.3f}  onsets warned {ew.get('detected_pct')}% lead {ew.get('median_lead_months')}")
    episodes = episode_evaluation(feat, probs, split["test_start"])
    if episodes:
        print("[M4] injected-episode detection:", json.dumps(episodes))

    # LightGBM feature importance (TreeSHAP) for the report
    import shap
    sample = feat.loc[te, ALL_FEATURES].sample(min(2000, int(te.sum())), random_state=RANDOM_SEED)
    sv = shap.TreeExplainer(lgbm_models["radview_lgbm"]).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    lgbm_importance = pd.Series(np.abs(sv).mean(0), index=ALL_FEATURES).sort_values(ascending=False)

    # ---------------- persist --------------------------------------------------
    base_rate = float(y[rows_tr].mean())
    torch.save({"state_dict": gru.state_dict(), "n_seeds": N_SEEDS, "calibration": calibs["radview_gru"],
                "base_rate": base_rate,
                "features": ALL_FEATURES, "mean": mean.to_dict(), "std": std.to_dict()}, MODELS_DIR / "m4_gru.pt")
    lgbm_models["radview_lgbm"].booster_.save_model(str(MODELS_DIR / "m4_lgbm.txt"))
    tp = test_frame[["business_id", "month", "decline"]].copy()
    for name, p in probs.items():
        if name != "majority":
            tp[name] = p[rows_te]
    tp.to_parquet(OUTPUTS_DIR / "m4_test_predictions.parquet", index=False)

    split.pop("test_start")
    report = {"split": split, "results": results, "injected_episodes": episodes, "gru_training": train_info,
              "calibration": calibs["radview_gru"], "primary_model": "radview_gru",
              "lgbm_top_features_shap": lgbm_importance.head(15).round(4).to_dict()}
    agreement = score(feat, gru, data, rows_tr, calibs["radview_gru"], base_rate)
    report["attribution_agreement"] = agreement
    (OUTPUTS_DIR / "m4_evaluation.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------
# Scoring and explanation
# --------------------------------------------------------------------------
def score(feat: pd.DataFrame, gru, data: SequenceData, background_rows: np.ndarray,
          calib: tuple[float, float], base_rate: float) -> dict:
    """Risk for every business-month; for each restaurant's latest month also SHAP, counterfactual
    attribution and star cost per aspect. Returns SHAP-vs-counterfactual agreement statistics."""
    all_rows = np.arange(len(feat))
    hist = feat[["business_id", "month", "decline", "rating_3m", "n_reviews_3m"]].copy()
    hist["risk"] = apply_platt(predict(gru, data, all_rows), calib)
    hist.to_parquet(OUTPUTS_DIR / "risk_history.parquet", index=False)

    last_month = feat["month"].max()
    latest_rows = np.flatnonzero((feat["month"] == last_month).to_numpy())
    print(f"[M4] explaining {len(latest_rows)} latest-month predictions (SHAP + counterfactual) ...")
    cols = ["business_id", "month", "rating_3m", "rating_6m", "rating_cum", "n_reviews_3m", "market_rating_3m"]
    cols += [f"{a}_{s}" for a in ASPECTS for s in ("neg_rate_3m", "market_rate")]
    latest = feat.loc[latest_rows, cols].reset_index(drop=True)
    latest["risk"] = hist["risk"].to_numpy()[latest_rows]
    latest["base_rate"] = base_rate
    latest["risk_lift"] = latest["risk"] / base_rate
    latest["risk_level"] = np.select([latest["risk_lift"] >= RISK_HIGH_LIFT, latest["risk_lift"] >= RISK_MEDIUM_LIFT],
                                     ["high", "medium"], "low")

    # SHAP (expected gradients) -> per feature group
    attr_df = pd.DataFrame(explain(gru, data, latest_rows, background_rows), columns=ALL_FEATURES)
    for group, gcols in FEATURE_GROUPS.items():
        latest[f"attr::{group}"] = attr_df[gcols].sum(axis=1).to_numpy()
    latest["top_features"] = [json.dumps(r.sort_values(key=np.abs, ascending=False).head(8).round(4).to_dict())
                              for _, r in attr_df.iterrows()]

    # Counterfactual: same GRU, one aspect reset to the market norm
    def predict_fn(frame, rows):
        cf_data = SequenceData(frame, ALL_FEATURES, data.mean, data.std)
        return apply_platt(predict(gru, cf_data, rows), calib)
    cf = counterfactual_attribution(feat, latest_rows, latest["risk"].to_numpy(), predict_fn)
    coefs = fit_star_impact(pd.read_parquet(PROCESSED_DIR / "panel_monthly.parquet"))
    stars = star_costs(latest, coefs)
    for a in ASPECTS:
        latest[f"cf::{a}"] = cf[a].to_numpy()
        latest[f"starcost::{a}"] = stars[a].to_numpy()
    (OUTPUTS_DIR / "m4_star_impact.json").write_text(json.dumps(coefs, indent=2))

    shap_aspects = latest[[f"attr::{a}" for a in ASPECTS]].set_axis(ASPECTS, axis=1)
    latest["top_aspect"] = shap_aspects.idxmax(axis=1)
    latest["top_aspect_cf"] = cf.idxmax(axis=1).to_numpy()
    latest.to_parquet(OUTPUTS_DIR / "risk_latest.parquet", index=False)

    at_risk = (latest["risk_level"] != "low").to_numpy()
    agreement = {"at_risk": method_agreement(shap_aspects, cf, at_risk),
                 "all": method_agreement(shap_aspects, cf, np.ones(len(latest), bool)),
                 "star_impact_coefficients": coefs}
    print(f"[M4] latest month {last_month}: {int((latest['risk_level'] == 'high').sum())} high-risk restaurants; "
          f"SHAP/counterfactual top-aspect agreement {agreement['all'].get('top_aspect_agreement')}")
    return agreement


def score_saved() -> dict:
    """Re-score with the saved GRU (used by the weekly CronJob - no retraining)."""
    ckpt = torch.load(MODELS_DIR / "m4_gru.pt", weights_only=False, map_location=DEVICE)
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet").sort_values(["business_id", "month"]).reset_index(drop=True)
    data = SequenceData(feat, ckpt["features"], pd.Series(ckpt["mean"]), pd.Series(ckpt["std"]))
    gru = GRUEnsemble([DeclineGRU(data.n_inputs) for _ in range(ckpt["n_seeds"])])
    gru.load_state_dict(ckpt["state_dict"])
    gru.to(DEVICE).eval()
    rng = np.random.default_rng(RANDOM_SEED)
    base_rate = ckpt.get("base_rate", float(feat["decline"].mean()))
    return score(feat, gru, data, rng.choice(len(feat), min(5000, len(feat)), replace=False),
                 tuple(ckpt["calibration"]), base_rate)


if __name__ == "__main__":
    run()
