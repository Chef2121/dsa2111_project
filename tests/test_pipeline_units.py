"""Unit tests for the core building blocks (fast; no trained models required)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from radview.config import ASPECTS, LABEL_NEG, LABEL_NONE, LABEL_POS


# ---------------------------------------------------------------- lexicon / M2 weak labels
def test_weak_labels_use_sentence_polarity_and_negation():
    from radview.m2_aspects.weak_labels import label_review

    lab = label_review("The food was delicious. Our server was rude and slow.", stars=3)
    assert lab["food"] == LABEL_POS
    assert lab["service"] == LABEL_NEG
    assert lab["cleanliness"] == LABEL_NONE

    # negation flips polarity: "not clean" is a complaint
    assert label_review("The bathroom was not clean at all.", stars=4)["cleanliness"] == LABEL_NEG


def test_weak_labels_fall_back_to_stars_only_when_known():
    from radview.m2_aspects.weak_labels import label_review

    text = "We tried the pasta."
    assert label_review(text, stars=5)["food"] == LABEL_POS
    assert label_review(text, stars=1)["food"] == LABEL_NEG
    assert label_review(text, stars=float("nan"))["food"] == LABEL_NONE  # unknown rating -> no evidence


def test_generic_words_are_not_aspect_terms():
    from radview.m2_aspects.weak_labels import label_review

    lab = label_review("Ordered dinner here for my birthday.", stars=1)
    assert all(v == LABEL_NONE for v in lab.values())


# ---------------------------------------------------------------- M1 ingestion
@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from radview.m1_ingest.ingest import run as ingest
    from radview.m1_ingest.synthetic import generate

    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("processed")
    generate(n_restaurants=6, seed=7, out_dir=raw)
    report = ingest(raw_dir=raw, out_dir=out)
    return raw, out, report


def test_ingest_filters_restaurants_and_deduplicates(tiny_dataset):
    raw, out, report = tiny_dataset
    biz = pd.read_parquet(out / "businesses.parquet")
    rev = pd.read_parquet(out / "reviews.parquet")
    assert report["restaurants"] == 6                       # non-restaurant businesses dropped
    assert biz["categories"].str.contains("Restaurants").all()
    assert rev["review_id"].is_unique
    assert not rev.duplicated(["user_id", "business_id", "text"]).any()
    assert set(rev["business_id"]) <= set(biz["business_id"])
    assert rev["month"].str.match(r"^\d{4}-\d{2}$").all()


# ---------------------------------------------------------------- M3 detector
def _panel(rates: list[float], n: int = 20) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(rates):
        row = {"business_id": "b1", "month": str(pd.Period("2020-01", "M") + i), "n_reviews": n,
               "sum_stars": 4.0 * n}
        for a in ASPECTS:
            row[f"{a}_neg"] = int(round(r * n)) if a == "service" else 1
        rows.append(row)
    return pd.DataFrame(rows)


def test_detector_flags_a_sudden_complaint_spike_only_after_it_happens(monkeypatch, tmp_path):
    import radview.m3_trends.detector as det

    monkeypatch.setattr(det, "PROCESSED_DIR", tmp_path)
    rates = [0.10] * 14 + [0.55, 0.60]
    out = det.detect(_panel(rates), save=False)
    assert out["service_spike"].iloc[:14].sum() == 0       # stable history -> no alarms
    assert out["service_spike"].iloc[14:].sum() >= 1       # jump is flagged
    assert out["food_spike"].sum() == 0                    # other aspects untouched


# ---------------------------------------------------------------- M4 features & sequences
def test_features_do_not_leak_future_data():
    """Changing months after t must not change any feature at t (only the label)."""
    from radview.m3_trends.detector import detect
    from radview.m4_forecast.features import ALL_FEATURES, build_features

    def panel(future_star: float):
        p = _panel([0.1] * 18)
        p["sumsq_stars"] = 16.0 * p["n_reviews"]
        for a in ASPECTS:
            p[f"{a}_pos"] = 2
        for col in ("n_low_star", "any_neg", "checkins", "tips"):
            p[col] = 1
        p.loc[15:, "sum_stars"] = future_star * p.loc[15:, "n_reviews"]
        p["mean_stars"] = p["sum_stars"] / p["n_reviews"]
        return p

    a, b = panel(4.0), panel(1.5)
    fa = build_features(a, detect(a, save=False))
    fb = build_features(b, detect(b, save=False))
    t = 14
    pd.testing.assert_series_equal(fa.loc[t, ALL_FEATURES], fb.loc[t, ALL_FEATURES], check_names=False)
    assert fa.loc[t, "decline"] == 0 and fb.loc[t, "decline"] == 1


def test_sequence_windows_are_left_padded_per_business():
    from radview.m4_forecast.model import SEQ_LEN, SequenceData

    frame = pd.DataFrame({"business_id": ["a"] * 3 + ["b"] * 2, "x": [1.0, 2, 3, 10, 20]})
    data = SequenceData(frame, ["x"], frame[["x"]].mean(), frame[["x"]].std())
    w = data.windows(np.array([2, 3]))
    assert w.shape == (2, SEQ_LEN, 2)
    assert w[0, -1, -1] == 0 and w[0, -4, -1] == 1          # padding flag before the 3 real months
    # business "b"'s first window must not see business "a"'s months
    assert w[1, :-1, -1].eq(1).all()


def test_platt_calibration_is_monotone():
    from radview.m4_forecast.model import apply_platt, fit_platt

    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < p * 0.3).astype(int)       # raw scores are 3x too high
    cal = fit_platt(p, y)
    q = apply_platt(np.sort(p), cal)
    assert np.all(np.diff(q) >= 0)
    assert q.mean() < p.mean()


# ---------------------------------------------------------------- M5 verifier
class _FakeStore:
    def __init__(self):
        self.reviews = pd.DataFrame(
            {"business_id": ["b1", "b2"], "text": ["Great pasta. Our server was rude and ignored us.", "Lovely."],
             **{f"{a}_label": [LABEL_NEG if a == "service" else 0, 0] for a in ASPECTS}},
            index=["r1", "r2"])


def test_verifier_accepts_grounded_and_rejects_ungrounded_claims(monkeypatch):
    import radview.m5_agent.verifier as ver

    monkeypatch.setattr(ver, "is_complaint", lambda sent, aspect: "rude" in sent)
    store = _FakeStore()
    metrics = {"rates": {"complaint_rate_3m": 0.25, "complaint_rate_prev6m": 0.05}, "stars": {"star_cost": 0.12}}
    ok = {"type": "review", "text": 'A guest says "Our server was rude and ignored us."',
          "review_ids": ["r1"], "quote": "Our server was rude and ignored us."}
    assert ver.verify_claim(ok, "b1", "service", metrics, store)["verified"]

    cases = [
        {**ok, "quote": "The waiter spat in my food."},                 # not in the review
        {**ok, "review_ids": ["r2"], "quote": "Lovely."},                # other restaurant & not a complaint
        {**ok, "quote": "Great pasta."},                                 # verbatim but not a service complaint
        {"type": "metric", "text": "60% of reviews complained about service."},  # unsupported figure
    ]
    for c in cases:
        assert not ver.verify_claim(c, "b1", "service", metrics, store)["verified"], c
    assert ver.verify_claim({"type": "metric", "text": "25% of reviews complained, up from 5%."},
                            "b1", "service", metrics, store)["verified"]


def test_verifier_checks_star_figures_substance_and_retrieval(monkeypatch):
    import radview.m5_agent.verifier as ver

    monkeypatch.setattr(ver, "is_complaint", lambda sent, aspect: "rude" in sent)
    store = _FakeStore()
    metrics = {"rates": {"complaint_rate_3m": 0.25}, "stars": {"star_cost": 0.12}}

    def check(claim, retrieved=None):
        return ver.verify_claim(claim, "b1", "service", metrics, store, retrieved)

    assert check({"type": "metric", "text": "This costs about 0.12 stars."})["verified"]
    assert check({"type": "metric", "text": "This costs about 0.90 stars."})["failed_axes"] == ["statistic"]
    assert check({"type": "metric", "text": "Service is the big problem."})["failed_axes"] == ["substance"]
    grounded = {"type": "review", "text": "x", "review_ids": ["r1"], "quote": "Our server was rude and ignored us."}
    assert check(grounded, retrieved={"r1"})["verified"]
    assert check(grounded, retrieved=set())["failed_axes"] == ["citation"]   # real, but never retrieved
    # figures inside a verbatim quote are the reviewer's, not the agent's
    assert check({**grounded, "text": 'They said "Our server was rude and ignored us." (99%)', "quote":
                  "Our server was rude and ignored us."})["failed_axes"] == ["statistic"]


def test_playbook_actions_follow_detected_sub_themes():
    from radview.m5_agent.playbook import actions_for, detect_sub_themes

    themes = detect_sub_themes("service", ["We waited 45 minutes.", "Service was painfully slow."])
    assert themes[0] == "wait times"
    acts = actions_for("service", themes)
    assert 1 <= len(acts) <= 3 and "staffing" in acts[0]
    json.dumps(acts)


# ---------------------------------------------------------------- M4 attribution
def test_star_impact_recovers_a_known_cost():
    from radview.m4_forecast.attribution import fit_star_impact

    rng = np.random.default_rng(1)
    n = 400
    rates = {a: rng.uniform(0, 0.5, n) for a in ASPECTS}
    stars = 4.5 - 2.0 * rates["cleanliness"] - 0.5 * rates["price"] + rng.normal(0, 0.05, n)
    panel = pd.DataFrame({"n_reviews": 20, "mean_stars": stars, **{f"{a}_neg": rates[a] * 20 for a in ASPECTS}})
    coefs = fit_star_impact(panel)
    assert coefs["cleanliness"] < coefs["price"] < 0          # bigger cost for cleanliness
    assert abs(coefs["food"]) < 0.3


def test_typical_aspect_frame_resets_only_that_aspect():
    from radview.m4_forecast.attribution import typical_aspect_frame

    feat = pd.DataFrame({f"{a}_{s}": [0.4] for a in ASPECTS
                         for s in ("neg_rate_3m", "neg_rate_change", "neg_rate_vs_market", "z", "spike_3m",
                                   "market_rate")} | {"n_spiking_aspects": [2]})
    for a in ASPECTS:
        feat[f"{a}_market_rate"] = 0.1
    cf = typical_aspect_frame(feat, "service")
    assert cf.loc[0, "service_neg_rate_3m"] == 0.1 and cf.loc[0, "service_z"] == 0
    assert cf.loc[0, "food_neg_rate_3m"] == 0.4                # other aspects untouched
