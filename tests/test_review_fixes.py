"""Tests for the fixes from the code review (cuisine, lexicon, M3 baseline, M4 split, verifier, cache)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from radview.config import ASPECTS, HORIZON_MONTHS, LABEL_NEG, LABEL_NONE, LABEL_POS


# ---------------------------------------------------------------- M1
def test_cuisine_skips_venue_and_drink_tags():
    """A restaurant's cuisine is its most common food category; venue and drink tags (Nightlife, Bars) are skipped."""
    from radview.m1_ingest.ingest import assign_cuisine

    cats = pd.Series(["Nightlife, Bars, Middle Eastern, Restaurants",
                      "Bars, Restaurants, Italian, Pizza",
                      "Restaurants, Italian",
                      "Nightlife, Bars, Restaurants"])
    assert assign_cuisine(cats).tolist() == ["Middle Eastern", "Italian", "Italian", "Restaurant"]


def test_early_selection_counts_only_the_first_year(tmp_path):
    """The survivor-free selection counts only the selected restaurants' reviews from the study's first year."""
    from radview.m1_ingest.ingest import REVIEW_FILE, early_review_counts

    rows = [("a", "2016-03-01"), ("a", "2016-11-30"), ("a", "2018-01-01"), ("b", "2017-02-01"), ("c", "2016-05-05")]
    (tmp_path / REVIEW_FILE).write_text("\n".join(
        json.dumps({"review_id": str(i), "business_id": b, "date": f"{d} 10:00:00"}) for i, (b, d) in enumerate(rows)))
    counts = early_review_counts(tmp_path, {"a", "b"}, "2016-01-01")
    assert counts == {"a": 2}                       # 2018 and 2017-02 are outside the year; "c" not selected


# ---------------------------------------------------------------- M2 weak labels
def test_idioms_and_negation_breakers():
    """Idioms such as 'can't wait to come back' and 'worth the wait' are not service mentions, 'nothing but great'
    is praise and 'not good' is a complaint.
    """
    from radview.lexicon import ASPECT_TERMS, aspect_tokens, sentence_polarity, tokenize

    def aspects(s):
        """Aspects whose seed terms appear in sentence ``s``."""
        return [a for a in ASPECTS if set(aspect_tokens(s)) & ASPECT_TERMS[a]]

    assert aspects("I can't wait to come back!") == []
    assert aspects("Worth the wait.") == []
    assert aspects("We had to wait 45 minutes.") == ["service"]
    assert sentence_polarity(tokenize("Nothing but great things to say.")) == 1
    assert sentence_polarity(tokenize("The service was not good.")) == -1


def test_three_star_reviews_do_not_lend_polarity():
    """Only 1-2 and 4-5 star reviews lend polarity to opinion-less sentences; a mixed 3-star review gives none."""
    from radview.m2_aspects.weak_labels import label_review

    neutral_words = "The service tonight."          # aspect word, no sentiment word
    assert label_review(neutral_words, 3.0)["service"] == LABEL_NONE
    assert label_review(neutral_words, 2.0)["service"] == LABEL_NEG
    assert label_review(neutral_words, 5.0)["service"] == LABEL_POS


# ---------------------------------------------------------------- M3
def test_baseline_is_the_true_median_and_mad_of_past_months():
    """The M3 baseline is exactly the median and median absolute deviation of the previous months, and undefined
    before six months of history.
    """
    from radview.m3_trends.detector import _baseline

    v = np.array([0.1, 0.3, 0.2, 0.5, 0.1, 0.4, 0.9, 0.2])
    med, mad = _baseline(v)
    assert np.isnan(med[:6]).all()                  # fewer than MIN_HISTORY past months
    past = v[:6]
    assert med[6] == np.median(past) and mad[6] == np.median(np.abs(past - np.median(past)))
    past = v[:7]
    assert med[7] == np.median(past) and mad[7] == np.median(np.abs(past - np.median(past)))


def test_m3_evaluation_reports_a_random_timing_baseline():
    """M3 evaluation reports a random-timing chance level (lift near 1 for random flags) and the lasting-drop
    metric.
    """
    from radview.m3_trends.detector import evaluate

    rng = np.random.default_rng(0)
    rows = []
    for b in range(30):
        for i in range(24):
            rows.append({"business_id": f"b{b}", "month": str(pd.Period("2018-01", "M") + i),
                         "decline": float(rng.random() < 0.1), "rating_6m": 4.0, "future_rating": 4.0,
                         "n_spiking_aspects": int(rng.random() < 0.1), "rating_alarm": 0})
    df = pd.DataFrame(rows)
    res = evaluate(df[["business_id", "month", "n_spiking_aspects", "rating_alarm"]],
                   df[["business_id", "month", "decline", "rating_6m", "future_rating"]])
    chance = res["complaint_spike"]["chance"]
    assert {"lift", "onsets_detected_pct", "lasting_drop_lift"} <= set(chance)
    assert 0.5 < chance["lift"] < 1.5                # random flags carry no information
    assert "lasting_drop" in res["complaint_spike"]


# ---------------------------------------------------------------- M4
def _labelled_frame(n_months: int = 48) -> pd.DataFrame:
    """A 48-month, one-restaurant frame with a decline label every seventh month."""
    months = [str(pd.Period("2016-01", "M") + i) for i in range(n_months)]
    return pd.DataFrame({"business_id": "b", "month": months, "decline": [float(i % 7 == 0) for i in range(n_months)]})


def test_split_purges_months_whose_label_reaches_the_next_period():
    """The time split leaves a three-month gap before validation and test, so no training label is computed from a
    later period.
    """
    from radview.m4_forecast.train import time_split

    feat = _labelled_frame()
    tr, va, te, info = time_split(feat)
    valid_start = feat.loc[va, "month"].min()
    test_start = feat.loc[te, "month"].min()
    last_train = pd.Period(feat.loc[tr, "month"].max(), "M")
    last_valid = pd.Period(feat.loc[va, "month"].max(), "M")
    assert last_train + HORIZON_MONTHS < pd.Period(valid_start, "M")   # label window ends before validation
    assert last_valid + HORIZON_MONTHS < pd.Period(test_start, "M")
    assert info["gap_months"] == HORIZON_MONTHS and not (tr & va).any() and not (va & te).any()


def test_early_warning_counts_calendar_months_and_reports_chance():
    """Early warning measures lead time in calendar months (missing labels do not shorten it) and reports the
    chance level of random alerts.
    """
    from radview.m4_forecast.train import early_warning

    # one business; onset at 2020-06 (0 in 2020-05, 1 in 2020-06); a label is missing in 2020-03
    months = ["2020-01", "2020-02", "2020-04", "2020-05", "2020-06"]
    frame = pd.DataFrame({"business_id": "b", "month": months, "decline": [0, 0, 0, 0, 1.0]})
    prob = np.array([0.0, 0.9, 0.0, 0.0, 0.0])     # one alert, in 2020-02
    ew = early_warning(frame, prob, threshold=0.5)
    assert ew["onsets"] == 1 and ew["detected_pct"] == 100.0
    assert ew["median_lead_months"] == 5            # 2020-02 -> start of the decline window 2020-07
    assert ew["chance_detected_pct"] is not None


def test_star_impact_never_rewards_complaints():
    """The star-cost model never gives complaints a positive effect, even when the data contain a spurious one."""
    from radview.m4_forecast.attribution import fit_star_impact

    rng = np.random.default_rng(2)
    n = 400
    rates = {a: rng.uniform(0, 0.5, n) for a in ASPECTS}
    stars = 4.5 - 1.5 * rates["food"] + 0.4 * rates["ambience"] + rng.normal(0, 0.05, n)  # spurious positive
    panel = pd.DataFrame({"n_reviews": 20, "mean_stars": stars, **{f"{a}_neg": rates[a] * 20 for a in ASPECTS}})
    coefs = fit_star_impact(panel)
    assert all(c <= 0 for c in coefs.values()) and coefs["ambience"] == 0 and coefs["food"] < -1


# ---------------------------------------------------------------- M5 verifier
class _Store:
    """One review of restaurant b1, labelled a service complaint."""
    def __init__(self):
        """Build the reviews table the verifier reads."""
        self.reviews = pd.DataFrame(
            {"business_id": ["b1"], "text": ["Our server was rude and ignored us."],
             **{f"{a}_label": [LABEL_NEG if a == "service" else 0] for a in ASPECTS}}, index=["r1"])


METRICS = {"rates": {"complaint_rate_3m": 0.12, "complaint_rate_prev6m": 0.05, "market_median_rate": 0.20,
                     "peer_median": 0.15, "risk": 0.09, "risk_drop_if_fixed": 0.02},
           "stars": {"star_cost": 0.12, "rating_3m": 3.88, "rating_6m": 4.05}}


def _check(monkeypatch, claim):
    """Verify ``claim`` for b1's service aspect against METRICS, treating only sentences with 'rude' as complaints."""
    import radview.m5_agent.verifier as ver

    monkeypatch.setattr(ver, "is_complaint", lambda sent, aspect: "rude" in sent)
    return ver.verify_claim(claim, "b1", "service", METRICS, _Store(), {"r1"})


def test_verifier_accepts_the_rule_based_wording(monkeypatch):
    """The rule-based planner's exact sentences pass, including the market median and the decline risk in the right
    clauses.
    """
    text = ("12 of 100 reviews in the last 3 months (12%) complained about service, versus 5% in the six months "
            "before; the typical restaurant in your market sits at 20%.")
    assert _check(monkeypatch, {"type": "metric", "text": text})["verified"]
    cost = "Compared with the typical restaurant in your market, service complaints are estimated to cost about 0.12 stars."
    assert _check(monkeypatch, {"type": "metric", "text": cost})["verified"]
    assert _check(monkeypatch, {"type": "metric", "text": "The decline risk is 9%."})["verified"]


def test_verifier_rejects_real_figures_attached_to_the_wrong_quantity(monkeypatch):
    """Real figures attached to the wrong quantity (market median, risk or rating used as something else) fail the
    statistic check.
    """
    for text in ("20% of your reviews complain about service.",       # market median as own rate
                 "9% of your reviews complain about service.",        # decline risk as complaint rate
                 "Service problems are costing you 3.88 stars."):     # current rating as star cost
        assert _check(monkeypatch, {"type": "metric", "text": text})["failed_axes"] == ["statistic"], text


def test_verifier_rejects_empty_quotes_and_multipliers(monkeypatch):
    """A review claim without a quote fails the quotation check, and an uncheckable multiplier such as 'tripled'
    fails the statistic check.
    """
    no_quote = {"type": "review", "text": "A guest found a cockroach in the soup.", "review_ids": ["r1"], "quote": ""}
    assert "quotation" in _check(monkeypatch, no_quote)["failed_axes"]
    quote = "Our server was rude and ignored us."
    tripled = {"type": "review", "text": f'Service complaints have tripled: "{quote}"', "review_ids": ["r1"],
               "quote": quote}
    assert _check(monkeypatch, tripled)["failed_axes"] == ["statistic"]


def test_free_text_figures_are_checked():
    """Figures in summaries and headlines must match a returned value; the risk lift may appear as a multiplier."""
    from radview.m5_agent.verifier import unsupported_text_figures

    assert unsupported_text_figures("Risk is 9%, 1.6x the typical restaurant.", [METRICS], 1.6) == []
    assert unsupported_text_figures("Complaints doubled to 45%.", [METRICS], 1.6) == ["45%", "doubled"]


# ---------------------------------------------------------------- M6 cache
def test_store_reloads_after_a_new_pipeline_run(tmp_path, monkeypatch):
    """The service cache reloads only when pipeline_run.json changes, so a long-running API picks up a rescore
    without a restart.
    """
    import os

    from radview.m5_agent import tools

    monkeypatch.setattr(tools, "OUTPUTS_DIR", tmp_path)
    store = tools.Store()
    marker = tmp_path / "pipeline_run.json"
    marker.write_text("{}")
    store.refresh_if_changed()
    store.__dict__["features"] = "old"               # as if cached_property had loaded it
    assert not store.refresh_if_changed() and store.__dict__["features"] == "old"
    os.utime(marker, (marker.stat().st_atime, marker.stat().st_mtime + 60))
    assert store.refresh_if_changed() and "features" not in store.__dict__
