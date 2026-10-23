"""Read-side service layer shared by the REST API (M6) and the dashboard.

All functions return plain JSON-serialisable Python objects so they can be
returned directly by FastAPI or consumed by Streamlit in-process.
"""
from __future__ import annotations

import csv
import functools
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

from radview.config import ALLOW_LLM_REFRESH, ASPECTS, LABEL_NEG, LABEL_POS, OUTPUTS_DIR, SOURCE
from radview.m5_agent.agent import PLANS_DIR, generate_plan
from radview.m5_agent.tools import STORE, compare_peers, query_metrics, run_forecast

FEEDBACK_FILE = OUTPUTS_DIR / "feedback.csv"


def fresh(fn):
    """Reload the cached pipeline outputs first if a newer pipeline run has finished."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        STORE.refresh_if_changed()
        return fn(*args, **kwargs)
    return wrapper


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


@fresh
def meta() -> dict:
    run = OUTPUTS_DIR / "pipeline_run.json"
    return _clean({"source": SOURCE, "latest_month": STORE.latest_month, "restaurants": int(len(STORE.businesses)),
                   "reviews": int(len(STORE.reviews)),
                   "pipeline_run": json.loads(run.read_text()) if run.exists() else None})


@fresh
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
    cols = ["business_id", "name", "city", "cuisine", "risk", "risk_lift", "risk_level", "rating_3m", "rating_6m", "rating_cum",
            "reviews_3m", "top_aspect", "spiking_aspects"] + [c for c in df.columns if c.startswith("attr::")]
    return _clean(df[cols].to_dict(orient="records"))


@fresh
def restaurant(business_id: str) -> dict:
    biz = STORE.businesses.loc[business_id]
    info = {k: biz[k] for k in ("name", "address", "city", "state", "cuisine", "categories", "yelp_stars",
                                "yelp_review_count", "is_open")}
    return _clean({"business_id": business_id, "info": info, "metrics": query_metrics(business_id),
                   "forecast": run_forecast(business_id), "peers": compare_peers(business_id)})


@fresh
def timeline(business_id: str) -> list[dict]:
    p = STORE.panel[STORE.panel["business_id"] == business_id][["month", "n_reviews", "mean_stars", "checkins"]]
    t = STORE.trends[STORE.trends["business_id"] == business_id]
    cols = ["month"] + [f"{a}_{s}" for a in ASPECTS for s in ("rate", "z", "spike")] + ["rating_alarm"]
    r = STORE.risk_history[STORE.risk_history["business_id"] == business_id][["month", "risk", "decline", "rating_3m"]]
    fcols = ["month"] + [f"{a}_{s}" for a in ASPECTS for s in ("neg_rate_3m", "market_rate")]
    f = STORE.features[STORE.features["business_id"] == business_id][fcols]
    df = p.merge(t[cols], on="month").merge(r, on="month", how="left").merge(f, on="month", how="left")
    return _clean(df.to_dict(orient="records"))


@fresh
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


@fresh
def plan(business_id: str, refresh: bool = False, use_llm: bool | None = None) -> dict:
    path = PLANS_DIR / f"{business_id}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    if not ALLOW_LLM_REFRESH:  # a request must not be able to trigger paid LLM calls unless explicitly allowed
        use_llm = False
    return _clean(generate_plan(business_id, use_llm=use_llm))


def evaluation() -> dict:
    out = {}
    for key in ("m2", "m3", "m4", "m5"):
        f = OUTPUTS_DIR / f"{key}_evaluation.json"
        out[key] = json.loads(f.read_text()) if f.exists() else None
    comp = OUTPUTS_DIR / "m5_planner_comparison.json"
    out["m5_comparison"] = json.loads(comp.read_text()) if comp.exists() else None
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


@fresh
def alerts(min_z: float = 0.0) -> list[dict]:
    """Live complaint-spike alerts (M3) for the latest month, most severe first."""
    t = STORE.trends[STORE.trends["month"] == STORE.latest_month].set_index("business_id")
    f = STORE.features[STORE.features["month"] == STORE.latest_month].set_index("business_id")
    names = STORE.businesses["name"]
    out = []
    for bid, row in t.iterrows():
        for a in ASPECTS:
            if not row[f"{a}_spike"] or row[f"{a}_z"] < min_z:
                continue
            z = float(row[f"{a}_z"])
            out.append({"business_id": bid, "name": names.get(bid, bid), "aspect": a, "robust_z": round(z, 2),
                        "complaint_rate_3m": float(f.loc[bid, f"{a}_neg_rate_3m"]),
                        "market_median_rate": float(f.loc[bid, f"{a}_market_rate"]),
                        "severity": "critical" if z >= 5 else "serious" if z >= 3.5 else "warning",
                        "risk": float(STORE.risk_latest.loc[bid, "risk"]) if bid in STORE.risk_latest.index else None})
    return _clean(sorted(out, key=lambda r: -r["robust_z"]))


@fresh
def analyse(texts: list[str]) -> list[dict]:
    """Score raw review texts on the fly with the production M2 model (e.g. reviews not yet ingested).

    Returns per-review aspect labels and, for each complaint, the sentence that carries it.
    """
    from radview.lexicon import split_sentences
    from radview.m5_agent.tools import aspect_model

    model = aspect_model()
    proba = model.predict_proba(texts, [float("nan")] * len(texts))
    names = {0: "not mentioned", LABEL_POS: "positive", LABEL_NEG: "negative"}
    out = []
    for text, p in zip(texts, proba):
        labels = {a: names[int(p[i].argmax())] for i, a in enumerate(ASPECTS)}
        evidence = {}
        complaints = [a for a, lab in labels.items() if lab == "negative"]
        if complaints:
            sents = split_sentences(text) or [text]
            sp = model.predict_proba(sents, [float("nan")] * len(sents))
            for a in complaints:
                i = ASPECTS.index(a)
                evidence[a] = sents[int(sp[:, i, LABEL_NEG].argmax())]
        out.append({"text": text, "aspects": labels,
                    "confidence": {a: round(float(p[i].max()), 3) for i, a in enumerate(ASPECTS)},
                    "complaint_sentences": evidence})
    return out
