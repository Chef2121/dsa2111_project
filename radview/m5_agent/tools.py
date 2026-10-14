"""M5 - tools the agent can call.

Each tool is a plain function over the artefacts written by M1-M4, so the same
functions back the LangGraph agent, the REST API and the dashboard:

    query_metrics(business_id)                rating & complaint trends, active spikes (M2/M3)
    run_forecast(business_id)                 decline risk + SHAP attribution (M4)
    get_evidence_reviews(business_id, aspect) recent negative reviews with verbatim quotes (M2)
    compare_peers(business_id)                complaint rates vs. same-cuisine restaurants in the city
"""
from __future__ import annotations

import json
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd

from radview.config import (ASPECTS, EVIDENCE_LOOKBACK_MONTHS, EVIDENCE_PER_ASPECT, LABEL_NEG, OUTPUTS_DIR,
                            PROCESSED_DIR)
from radview.lexicon import split_sentences


class Store:
    """Lazily loaded, process-wide cache of the pipeline outputs."""

    @cached_property
    def businesses(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED_DIR / "businesses.parquet").set_index("business_id")

    @cached_property
    def reviews(self) -> pd.DataFrame:
        rev = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
        asp = pd.read_parquet(PROCESSED_DIR / "review_aspects.parquet")
        return rev.merge(asp, on="review_id").set_index("review_id", drop=False)

    @cached_property
    def reviews_by_business(self) -> dict[str, np.ndarray]:
        return self.reviews.groupby("business_id").indices

    @cached_property
    def panel(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED_DIR / "panel_monthly.parquet")

    @cached_property
    def trends(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED_DIR / "aspect_trends.parquet")

    @cached_property
    def features(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED_DIR / "features.parquet")

    @cached_property
    def risk_latest(self) -> pd.DataFrame:
        return pd.read_parquet(OUTPUTS_DIR / "risk_latest.parquet").set_index("business_id")

    @cached_property
    def risk_history(self) -> pd.DataFrame:
        return pd.read_parquet(OUTPUTS_DIR / "risk_history.parquet")

    @cached_property
    def latest_month(self) -> str:
        return str(self.features["month"].max())

    def business_reviews(self, business_id: str) -> pd.DataFrame:
        idx = self.reviews_by_business.get(business_id)
        return self.reviews.iloc[idx] if idx is not None else self.reviews.iloc[0:0]


STORE = Store()


def _month_offset(month: str, k: int) -> str:
    return str(pd.Period(month, "M") + k)


def query_metrics(business_id: str, store: Store = STORE) -> dict:
    """Rating trend and per-aspect complaint statistics for the latest month."""
    f = store.features
    row = f[(f["business_id"] == business_id) & (f["month"] == store.latest_month)]
    if row.empty:
        raise KeyError(business_id)
    row = row.iloc[0]
    tr = store.trends
    trow = tr[(tr["business_id"] == business_id) & (tr["month"] == store.latest_month)].iloc[0]
    aspects = {}
    for a in ASPECTS:
        aspects[a] = {
            "complaint_rate_3m": round(float(row[f"{a}_neg_rate_3m"]), 4),
            "complaint_rate_prev6m": round(float(row[f"{a}_neg_rate_3m"] - row[f"{a}_neg_rate_change"]), 4),
            "change": round(float(row[f"{a}_neg_rate_change"]), 4),
            "praise_rate_3m": round(float(row[f"{a}_pos_rate_3m"]), 4),
            "robust_z": round(float(trow[f"{a}_z"]), 2),
            "spike": bool(trow[f"{a}_spike"]) or bool(row[f"{a}_spike_3m"]),
        }
    return {
        "business_id": business_id, "month": store.latest_month,
        "rating_3m": None if pd.isna(row["rating_3m"]) else round(float(row["rating_3m"]), 2),
        "rating_6m": None if pd.isna(row["rating_6m"]) else round(float(row["rating_6m"]), 2),
        "rating_all_time": None if pd.isna(row["rating_cum"]) else round(float(row["rating_cum"]), 2),
        "reviews_3m": int(row["n_reviews_3m"]),
        "aspects": aspects,
    }


def run_forecast(business_id: str, store: Store = STORE) -> dict:
    """Decline probability and the SHAP attribution of each driver group (logit units)."""
    r = store.risk_latest.loc[business_id]
    attributions = {c.split("::", 1)[1]: round(float(r[c]), 4) for c in r.index if c.startswith("attr::")}
    return {"business_id": business_id, "month": r["month"], "risk": round(float(r["risk"]), 4),
            "risk_level": r["risk_level"], "attributions": attributions,
            "top_features": json.loads(r["top_features"])}


@lru_cache(maxsize=1)
def aspect_model():
    """The production M2 model, loaded once per process (used for sentence-level checks)."""
    from radview.m2_aspects.models import load_best_model
    return load_best_model()


def sentence_complaint_scores(sentences: list[str], aspect: str) -> np.ndarray:
    """P(negative about ``aspect``) for each sentence, according to the M2 model."""
    if not sentences:
        return np.array([])
    proba = aspect_model().predict_proba(sentences, [float("nan")] * len(sentences))
    return proba[:, ASPECTS.index(aspect), LABEL_NEG]


def is_complaint(sentence: str, aspect: str) -> bool:
    """Does M2 classify this sentence on its own as a complaint about ``aspect``?"""
    proba = aspect_model().predict_proba([sentence], [float("nan")])[0, ASPECTS.index(aspect)]
    return int(proba.argmax()) == LABEL_NEG


def get_evidence_reviews(business_id: str, aspect: str, k: int = EVIDENCE_PER_ASPECT,
                         lookback_months: int = EVIDENCE_LOOKBACK_MONTHS, store: Store = STORE) -> list[dict]:
    """Recent reviews whose *quoted sentence* is itself a complaint about ``aspect``.

    1. candidate reviews: M2 labels the review negative on the aspect, ranked by
       confidence x recency (half-life ~4 months);
    2. each candidate is split into sentences and M2 re-scores every sentence;
       the most negative sentence becomes the quote, and the review is kept only
       if that sentence alone is classified as a complaint about the aspect.
    """
    rev = store.business_reviews(business_id)
    since = _month_offset(store.latest_month, -(lookback_months - 1))
    cand = rev[(rev["month"] >= since) & (rev[f"{aspect}_label"] == LABEL_NEG)]
    if len(cand) < k:  # widen the window once rather than return too little
        since = _month_offset(store.latest_month, -(2 * lookback_months - 1))
        cand = rev[(rev["month"] >= since) & (rev[f"{aspect}_label"] == LABEL_NEG)]
    if cand.empty:
        return []
    age = (pd.Timestamp(cand["date"].max()) - cand["date"]).dt.days.to_numpy()
    score = cand[f"{aspect}_neg_p"].to_numpy() * np.exp(-age / 120.0)
    cand = cand.iloc[np.argsort(-score)[:4 * k]]

    sents, owner = [], []
    for i, text in enumerate(cand["text"]):
        for s_ in split_sentences(text)[:30]:
            if 3 <= len(s_.split()) <= 60:
                sents.append(s_)
                owner.append(i)
    neg_p = sentence_complaint_scores(sents, aspect)
    best: dict[int, tuple[float, str]] = {}
    for i, s_, p in zip(owner, sents, neg_p):
        if p > best.get(i, (-1.0, ""))[0]:
            best[i] = (float(p), s_)
    out = []
    for i, r in enumerate(cand.itertuples()):
        if i not in best or not is_complaint(best[i][1], aspect):
            continue
        out.append({"review_id": r.review_id, "date": str(r.date.date()), "stars": float(r.stars), "aspect": aspect,
                    "confidence": round(best[i][0], 3), "quote": best[i][1], "text": r.text})
        if len(out) == k:
            break
    return out


def compare_peers(business_id: str, store: Store = STORE) -> dict:
    """Complaint rates vs. the median of same-cuisine (fallback: same-city) restaurants."""
    biz = store.businesses
    me = biz.loc[business_id]
    peers = biz[(biz["city"] == me["city"]) & (biz["cuisine"] == me["cuisine"])].index
    basis = f"{me['cuisine']} restaurants in {me['city']}"
    if len(peers) < 5:
        peers, basis = biz[biz["city"] == me["city"]].index, f"restaurants in {me['city']}"
    f = store.features
    cur = f[(f["month"] == store.latest_month) & f["business_id"].isin(peers) & (f["n_reviews_3m"] > 0)]
    out = {"basis": basis, "n_peers": int(len(cur)), "aspects": {}}
    mine = f[(f["month"] == store.latest_month) & (f["business_id"] == business_id)].iloc[0]
    for a in ASPECTS:
        med = float(cur[f"{a}_neg_rate_3m"].median()) if len(cur) else float("nan")
        out["aspects"][a] = {"mine": round(float(mine[f"{a}_neg_rate_3m"]), 4), "peer_median": round(med, 4),
                             "percentile": round(float((cur[f"{a}_neg_rate_3m"] < mine[f"{a}_neg_rate_3m"]).mean() * 100), 1)
                             if len(cur) else None}
    return out


TOOL_REGISTRY = {
    "query_metrics": query_metrics,
    "run_forecast": run_forecast,
    "get_evidence_reviews": get_evidence_reviews,
    "compare_peers": compare_peers,
}
