"""Read-side service layer shared by the REST API (M6) and the dashboard.

All functions return plain JSON-serialisable Python objects so they can be
returned directly by FastAPI or consumed by Streamlit in-process.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

from radview.config import ASPECTS, LABEL_NEG, LABEL_POS, OUTPUTS_DIR, SOURCE
from radview.m5_agent.agent import PLANS_DIR, generate_plan
from radview.m5_agent.tools import STORE, compare_peers, query_metrics, run_forecast

FEEDBACK_FILE = OUTPUTS_DIR / "feedback.csv"


def _clean(obj):
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def meta() -> dict:
    run = OUTPUTS_DIR / "pipeline_run.json"
    return _clean({"source": SOURCE, "latest_month": STORE.latest_month, "restaurants": int(len(STORE.businesses)),
                   "reviews": int(len(STORE.reviews)),
                   "pipeline_run": json.loads(run.read_text()) if run.exists() else None})


def list_restaurants(active_only: bool = True) -> list[dict]:
    latest = STORE.risk_latest.reset_index()
    feat = STORE.features[STORE.features["month"] == STORE.latest_month].set_index("business_id")
    trends = STORE.trends[STORE.trends["month"] == STORE.latest_month].set_index("business_id")
    biz = STORE.businesses
    df = latest.merge(biz[["name", "city", "cuisine", "is_open"]], left_on="business_id", right_index=True)
    df["reviews_3m"] = feat.loc[df["business_id"], "n_reviews_3m"].to_numpy()
    df["spiking_aspects"] = [[a for a in ASPECTS if trends.loc[b, f"{a}_spike"] or feat.loc[b, f"{a}_spike_3m"]]
                             for b in df["business_id"]]
    if active_only:
        df = df[df["reviews_3m"] > 0]
    df = df.sort_values("risk", ascending=False)
    cols = ["business_id", "name", "city", "cuisine", "risk", "risk_level", "rating_3m", "rating_6m", "rating_cum",
            "reviews_3m", "top_aspect", "spiking_aspects"] + [c for c in df.columns if c.startswith("attr::")]
    return _clean(df[cols].to_dict(orient="records"))


def restaurant(business_id: str) -> dict:
    biz = STORE.businesses.loc[business_id]
    info = {k: biz[k] for k in ("name", "address", "city", "state", "cuisine", "categories", "yelp_stars",
                                "yelp_review_count", "is_open")}
    return _clean({"business_id": business_id, "info": info, "metrics": query_metrics(business_id),
                   "forecast": run_forecast(business_id), "peers": compare_peers(business_id)})


def timeline(business_id: str) -> list[dict]:
    p = STORE.panel[STORE.panel["business_id"] == business_id][["month", "n_reviews", "mean_stars", "checkins"]]
    t = STORE.trends[STORE.trends["business_id"] == business_id]
    cols = ["month"] + [f"{a}_{s}" for a in ASPECTS for s in ("rate", "z", "spike")] + ["rating_alarm"]
    r = STORE.risk_history[STORE.risk_history["business_id"] == business_id][["month", "risk", "decline", "rating_3m"]]
    df = p.merge(t[cols], on="month").merge(r, on="month", how="left")
    return _clean(df.to_dict(orient="records"))


def reviews(business_id: str, aspect: str | None = None, sentiment: str | None = None, limit: int = 50) -> list[dict]:
    rev = STORE.business_reviews(business_id).sort_values("date", ascending=False)
    if aspect:
        want = {"negative": [LABEL_NEG], "positive": [LABEL_POS]}.get(sentiment or "", [LABEL_NEG, LABEL_POS])
        rev = rev[rev[f"{aspect}_label"].isin(want)]
    out = []
    for r in rev.head(limit).itertuples():
        out.append({"review_id": r.review_id, "date": str(r.date.date()), "stars": float(r.stars), "text": r.text,
                    "aspects": {a: int(getattr(r, f"{a}_label")) for a in ASPECTS}})
    return out


def plan(business_id: str, refresh: bool = False, use_llm: bool | None = None) -> dict:
    path = PLANS_DIR / f"{business_id}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    return _clean(generate_plan(business_id, use_llm=use_llm))


def evaluation() -> dict:
    out = {}
    for key in ("m2", "m3", "m4", "m5"):
        f = OUTPUTS_DIR / f"{key}_evaluation.json"
        out[key] = json.loads(f.read_text()) if f.exists() else None
    out["feedback"] = feedback_summary()
    return _clean(out)


def m4_curves(points: int = 60) -> dict:
    """PR and ROC curve points for every M4 model on the test period."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    f = OUTPUTS_DIR / "m4_test_predictions.parquet"
    if not f.exists():
        return {}
    tp = pd.read_parquet(f)
    y = tp["decline"].to_numpy()
    out = {}
    for m in [c for c in tp.columns if c not in {"business_id", "month", "decline"}]:
        prec, rec, _ = precision_recall_curve(y, tp[m])
        fpr, tpr, _ = roc_curve(y, tp[m])
        ip = np.linspace(0, len(rec) - 1, points).astype(int)
        ir = np.linspace(0, len(fpr) - 1, points).astype(int)
        out[m] = {"pr": {"recall": rec[ip].round(4).tolist(), "precision": prec[ip].round(4).tolist()},
                  "roc": {"fpr": fpr[ir].round(4).tolist(), "tpr": tpr[ir].round(4).tolist()}}
    return out


def add_feedback(business_id: str, usefulness: int, comment: str = "", aspect: str | None = None) -> dict:
    """Owner/human rating (1-5) of an action plan - the M5 usefulness metric."""
    if not 1 <= int(usefulness) <= 5:
        raise ValueError("usefulness must be 1-5")
    new = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "business_id", "aspect", "usefulness", "comment"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), business_id, aspect or "", int(usefulness), comment])
    return feedback_summary()


def feedback_summary() -> dict:
    if not FEEDBACK_FILE.exists():
        return {"n": 0, "mean_usefulness": None}
    fb = pd.read_csv(FEEDBACK_FILE)
    return {"n": int(len(fb)), "mean_usefulness": round(float(fb["usefulness"].mean()), 2) if len(fb) else None}
