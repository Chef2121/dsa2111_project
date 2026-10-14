"""M4 - train, compare and deploy the rating-decline forecasters.

Models compared on an out-of-time test period (no temporal leakage: the
split is by calendar month and every feature only uses past data):

    rating_logreg   logistic regression, rating-trend features only  (baseline)
    rating_lgbm     LightGBM, rating-trend features only               (baseline)
    sentiment_lgbm  LightGBM, rating + aspect-agnostic negativity      (baseline)
    radview_lgbm    LightGBM, all features incl. per-aspect M2/M3 signals
    radview_gru     PyTorch GRU over 12-month sequences of all features (Radview)

Metrics: AUROC, PR-AUC (the honest metric for a rare positive class),
precision/recall in the top 10% riskiest business-months, and early-warning
performance at a fixed alert budget (share of decline onsets flagged in the
preceding 6 months, and the median lead time in months).

Artefacts:
    models/m4_gru.pt, models/m4_lgbm.txt
    outputs/m4_evaluation.json, outputs/m4_test_predictions.parquet
    outputs/risk_history.parquet   risk score for every business-month
    outputs/risk_latest.parquet    latest risk + SHAP attribution by aspect

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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from radview.config import (ASPECTS, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, RANDOM_SEED, RISK_HIGH, RISK_MEDIUM,
                            TEST_MONTHS, VALID_MONTHS)
from radview.m4_forecast.features import ALL_FEATURES, RATING_FEATURES, SENTIMENT_FEATURES
from radview.m4_forecast.model import (GRUEnsemble, SequenceData, apply_platt, explain, fit_platt, predict,
                                      train_gru)

N_SEEDS = 3  # GRU ensemble size

ALERT_BUDGET = 0.15     # share of business-months each model may flag in the early-warning test
LOOKBACK_ALERT = 6      # months before an onset in which an alert counts

FEATURE_GROUPS = {
    **{a: [f for f in ALL_FEATURES if f.startswith(a + "_")] for a in ASPECTS},
    "rating trend": [f for f in RATING_FEATURES if f.startswith("rating_")],
    "volume & footfall": ["n_reviews_3m", "review_velocity_ratio", "months_active", "checkins_3m", "checkin_ratio", "tips_3m"],
    "overall sentiment": ["low_star_share_3m", "any_neg_share_3m", "any_neg_share_change", "n_spiking_aspects"],
}


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
            "test": f"{test_start}..{months[-1]}",
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
    top = np.argsort(-p)[:k]
    thr = np.quantile(p, 1 - ALERT_BUDGET)
    return {"auroc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
            "precision_at_10pct": round(float(y[top].mean()), 4),
            "recall_at_10pct": round(float(y[top].sum() / max(y.sum(), 1)), 4),
            "early_warning": early_warning(frame, p, thr)}


def run(epochs: int = 30) -> dict:
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet").sort_values(["business_id", "month"]).reset_index(drop=True)
    tr, va, te, split = time_split(feat)
    print("[M4] split:", split)
    y = feat["decline"].fillna(0).to_numpy().astype(int)
    preds, results = {}, {}

    # ---------------- baselines ----------------------------------------------
    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
    logreg.fit(feat.loc[tr, RATING_FEATURES], y[tr])
    preds["rating_logreg"] = logreg.predict_proba(feat.loc[te, RATING_FEATURES])[:, 1]

    lgbm_models = {}
    for name, cols in (("rating_lgbm", RATING_FEATURES), ("sentiment_lgbm", SENTIMENT_FEATURES),
                       ("radview_lgbm", ALL_FEATURES)):
        m = _lgbm(feat.loc[tr, cols], y[tr], feat.loc[va, cols], y[va])
        lgbm_models[name] = m
        preds[name] = m.predict_proba(feat.loc[te, cols])[:, 1]

    # ---------------- Radview GRU (PyTorch) ---------------------------------
    mean = feat.loc[tr, ALL_FEATURES].mean()
    std = feat.loc[tr, ALL_FEATURES].std().replace(0, 1).fillna(1)
    data = SequenceData(feat, ALL_FEATURES, mean, std)
    rows_tr, rows_va, rows_te = (np.flatnonzero(m.to_numpy()) for m in (tr, va, te))
    members, train_info = [], []
    for k in range(N_SEEDS):
        m, info = train_gru(data, y, rows_tr, rows_va, epochs=epochs, seed=RANDOM_SEED + k)
        members.append(m)
        train_info.append(info)
    gru = GRUEnsemble(members).eval()
    calib = fit_platt(predict(gru, data, rows_va), y[rows_va])
    preds["radview_gru"] = apply_platt(predict(gru, data, rows_te), calib)

    test_frame = feat.loc[te].reset_index(drop=True)
    for name, p in preds.items():
        results[name] = metrics(y[te], p, test_frame)
        r = results[name]
        print(f"[M4] {name:15s} AUROC {r['auroc']:.3f}  PR-AUC {r['pr_auc']:.3f}  "
              f"P@10% {r['precision_at_10pct']:.3f}  early-warning {r['early_warning']['detected_pct']}% "
              f"lead {r['early_warning']['median_lead_months']}")

    # LightGBM feature importance (SHAP) for the report
    import shap
    sample = feat.loc[te, ALL_FEATURES].sample(min(2000, int(te.sum())), random_state=RANDOM_SEED)
    sv = shap.TreeExplainer(lgbm_models["radview_lgbm"]).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    lgbm_importance = pd.Series(np.abs(sv).mean(0), index=ALL_FEATURES).sort_values(ascending=False)

    # ---------------- persist --------------------------------------------------
    torch.save({"state_dict": gru.state_dict(), "n_seeds": N_SEEDS, "calibration": calib, "features": ALL_FEATURES,
                "mean": mean.to_dict(), "std": std.to_dict()}, MODELS_DIR / "m4_gru.pt")
    lgbm_models["radview_lgbm"].booster_.save_model(str(MODELS_DIR / "m4_lgbm.txt"))
    tp = test_frame[["business_id", "month", "decline"]].copy()
    for name, p in preds.items():
        tp[name] = p
    tp.to_parquet(OUTPUTS_DIR / "m4_test_predictions.parquet", index=False)

    report = {"split": split, "results": results, "gru_training": train_info, "calibration": calib,
              "lgbm_top_features_shap": lgbm_importance.head(15).round(4).to_dict(), "primary_model": "radview_gru"}
    (OUTPUTS_DIR / "m4_evaluation.json").write_text(json.dumps(report, indent=2))

    score(feat, gru, data, rows_tr, calib)
    return report


def score(feat: pd.DataFrame, gru, data: SequenceData, background_rows: np.ndarray,
          calib: tuple[float, float]) -> pd.DataFrame:
    """Risk for every business-month + SHAP attribution for each business's latest month."""
    all_rows = np.arange(len(feat))
    hist = feat[["business_id", "month", "decline", "rating_3m", "n_reviews_3m"]].copy()
    hist["risk"] = apply_platt(predict(gru, data, all_rows), calib)
    hist.to_parquet(OUTPUTS_DIR / "risk_history.parquet", index=False)

    last_month = feat["month"].max()
    latest_rows = np.flatnonzero((feat["month"] == last_month).to_numpy())
    print(f"[M4] explaining {len(latest_rows)} latest-month predictions with SHAP ...")
    attrs = explain(gru, data, latest_rows, background_rows)
    latest = feat.loc[latest_rows, ["business_id", "month", "rating_3m", "rating_6m", "rating_cum", "n_reviews_3m"]].reset_index(drop=True)
    latest["risk"] = hist["risk"].to_numpy()[latest_rows]
    latest["risk_level"] = np.select([latest["risk"] >= RISK_HIGH, latest["risk"] >= RISK_MEDIUM], ["high", "medium"], "low")
    attr_df = pd.DataFrame(attrs, columns=ALL_FEATURES)
    for group, cols in FEATURE_GROUPS.items():
        latest[f"attr::{group}"] = attr_df[cols].sum(axis=1).to_numpy()
    latest["top_features"] = [json.dumps(r.sort_values(key=np.abs, ascending=False).head(8).round(4).to_dict())
                              for _, r in attr_df.iterrows()]
    aspect_cols = [f"attr::{a}" for a in ASPECTS]
    latest["top_aspect"] = latest[aspect_cols].idxmax(axis=1).str.replace("attr::", "", regex=False)
    latest.to_parquet(OUTPUTS_DIR / "risk_latest.parquet", index=False)
    print(f"[M4] latest month {last_month}: {int((latest['risk_level'] == 'high').sum())} high-risk restaurants")
    return latest



def score_saved() -> pd.DataFrame:
    """Re-score with the saved GRU (used by the weekly CronJob - no retraining)."""
    from radview.m4_forecast.model import DeclineGRU

    ckpt = torch.load(MODELS_DIR / "m4_gru.pt", weights_only=False)
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet").sort_values(["business_id", "month"]).reset_index(drop=True)
    data = SequenceData(feat, ckpt["features"], pd.Series(ckpt["mean"]), pd.Series(ckpt["std"]))
    gru = GRUEnsemble([DeclineGRU(data.n_inputs) for _ in range(ckpt["n_seeds"])])
    gru.load_state_dict(ckpt["state_dict"])
    gru.eval()
    rng = np.random.default_rng(RANDOM_SEED)
    return score(feat, gru, data, rng.choice(len(feat), min(5000, len(feat)), replace=False), tuple(ckpt["calibration"]))


if __name__ == "__main__":
    run()
