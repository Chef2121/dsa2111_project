"""M4 - feature engineering and decline labels.

Every feature for month t uses data up to and including t only. The label
looks *forward*:

    ref    = mean star rating over months t-5..t        (needs >= MIN_REVIEWS_PAST reviews)
    future = mean star rating over months t+1..t+3      (needs >= MIN_REVIEWS_FUTURE reviews)
    decline = 1  if  future <= ref - DECLINE_THRESHOLD
                 and (ref - future) / s.e. >= DECLINE_MIN_Z   (one-sided z-test)

The significance condition stops small-sample noise (regression to the mean
after a lucky month) from being labelled as a decline.

Market-relative features compare a restaurant with the median restaurant in
the same city in the *same* month (a cross-section at t, so still leak-free):
being 10 points above the market on service complaints means more than a raw
rate that may simply reflect the whole market's mood.

Feature groups (used to build the baselines):
    RATING_FEATURES    - rating level / trend / volume       -> "rating-trend" baseline
    SENTIMENT_FEATURES - + overall (aspect-agnostic) negativity -> "sentiment" baseline
    ASPECT_FEATURES    - per-aspect complaint rates, growth and M3 z-scores -> Radview
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from radview.config import (ASPECTS, DECLINE_MIN_Z, DECLINE_THRESHOLD, HORIZON_MONTHS, MIN_REVIEWS_FUTURE, MIN_REVIEWS_PAST,
                            REFERENCE_MONTHS)

RATING_FEATURES = ["rating_1m", "rating_3m", "rating_6m", "rating_cum", "rating_3m_vs_prev6m", "rating_slope_6m",
                   "rating_3m_vs_market", "n_reviews_3m", "review_velocity_ratio", "months_active", "checkins_3m",
                   "checkin_ratio", "tips_3m"]
SENTIMENT_FEATURES = RATING_FEATURES + ["low_star_share_3m", "any_neg_share_3m", "any_neg_share_change"]
ASPECT_FEATURES = [f"{a}_{s}" for a in ASPECTS
                   for s in ("neg_rate_3m", "neg_rate_change", "neg_rate_vs_market", "pos_rate_3m", "z", "spike_3m")]
ALL_FEATURES = SENTIMENT_FEATURES + ASPECT_FEATURES + ["n_spiking_aspects"]


def _roll(g, col, w):
    return g[col].transform(lambda s: s.rolling(w, min_periods=1).sum())


def _slope(y: np.ndarray) -> float:
    m = ~np.isnan(y)
    if m.sum() < 3:
        return 0.0
    x = np.arange(len(y))[m]
    return float(np.polyfit(x, y[m], 1)[0])


def build_features(panel: pd.DataFrame, trends: pd.DataFrame, cities: pd.Series | None = None) -> pd.DataFrame:
    """``cities`` maps business_id -> city (market). Without it all restaurants form one market."""
    p = panel.merge(trends, on=["business_id", "month"], how="left").sort_values(["business_id", "month"]).reset_index(drop=True)
    g = p.groupby("business_id", sort=False)
    f = p[["business_id", "month"]].copy()

    def win(col, w):
        return _roll(g, col, w)

    n1, n3, n6 = p["n_reviews"], win("n_reviews", 3), win("n_reviews", REFERENCE_MONTHS)
    s1, s3, s6 = p["sum_stars"], win("sum_stars", 3), win("sum_stars", REFERENCE_MONTHS)
    n_prev6 = g["n_reviews"].transform(lambda s: s.shift(3).rolling(6, min_periods=1).sum())
    s_prev6 = g["sum_stars"].transform(lambda s: s.shift(3).rolling(6, min_periods=1).sum())
    n_cum, s_cum = g["n_reviews"].cumsum(), g["sum_stars"].cumsum()

    f["rating_1m"] = (s1 / n1.where(n1 > 0))
    f["rating_3m"] = s3 / n3.where(n3 > 0)
    f["rating_6m"] = s6 / n6.where(n6 > 0)
    f["rating_cum"] = s_cum / n_cum.where(n_cum > 0)
    f["rating_3m_vs_prev6m"] = f["rating_3m"] - s_prev6 / n_prev6.where(n_prev6 > 0)
    f["rating_slope_6m"] = g["mean_stars"].transform(lambda s: s.rolling(6, min_periods=3).apply(_slope, raw=True))
    f["n_reviews_3m"] = n3
    f["review_velocity_ratio"] = n3 / (n_prev6 / 2).clip(lower=1)
    f["months_active"] = g.cumcount() + 1
    ck3 = win("checkins", 3)
    ck_prev = g["checkins"].transform(lambda s: s.shift(3).rolling(6, min_periods=1).sum())
    f["checkins_3m"] = ck3
    f["checkin_ratio"] = ck3 / (ck_prev / 2).clip(lower=1)
    f["tips_3m"] = win("tips", 3)

    f["low_star_share_3m"] = win("n_low_star", 3) / n3.clip(lower=1)
    any3 = win("any_neg", 3) / n3.clip(lower=1)
    any_prev = g["any_neg"].transform(lambda s: s.shift(3).rolling(6, min_periods=1).sum()) / n_prev6.clip(lower=1)
    f["any_neg_share_3m"] = any3
    f["any_neg_share_change"] = any3 - any_prev

    for a in ASPECTS:
        r3 = win(f"{a}_neg", 3) / n3.clip(lower=1)
        rprev = g[f"{a}_neg"].transform(lambda s: s.shift(3).rolling(6, min_periods=1).sum()) / n_prev6.clip(lower=1)
        f[f"{a}_neg_rate_3m"] = r3
        f[f"{a}_neg_rate_change"] = r3 - rprev
        f[f"{a}_pos_rate_3m"] = win(f"{a}_pos", 3) / n3.clip(lower=1)
        f[f"{a}_z"] = p[f"{a}_z"]
        f[f"{a}_spike_3m"] = g[f"{a}_spike"].transform(lambda s: s.rolling(3, min_periods=1).max())
    f["n_spiking_aspects"] = p["n_spiking_aspects"]

    # ---------------- market (same city, same month) comparisons ------------
    market = f["business_id"].map(cities) if cities is not None else pd.Series("all", index=f.index)
    market = market.fillna("unknown")
    active = f["n_reviews_3m"] > 0
    keys = [market, f["month"]]

    def market_median(col):
        return f[col].where(active).groupby(keys).transform("median")

    f["market_rating_3m"] = market_median("rating_3m")
    f["rating_3m_vs_market"] = f["rating_3m"] - f["market_rating_3m"]
    for a in ASPECTS:
        f[f"{a}_market_rate"] = market_median(f"{a}_neg_rate_3m")  # kept for attribution / dashboard
        f[f"{a}_neg_rate_vs_market"] = f[f"{a}_neg_rate_3m"] - f[f"{a}_market_rate"]

    # ---------------- forward-looking label ---------------------------------
    def fwd(col):  # sum over months t+1 .. t+HORIZON
        return g[col].transform(lambda s: s[::-1].rolling(HORIZON_MONTHS, min_periods=HORIZON_MONTHS).sum()[::-1].shift(-1))

    n_fut, s_fut, ss_fut = fwd("n_reviews"), fwd("sum_stars"), fwd("sumsq_stars")
    ss6 = win("sumsq_stars", REFERENCE_MONTHS)
    f["future_rating"] = s_fut / n_fut.where(n_fut > 0)
    f["ref_rating"] = f["rating_6m"]
    var_ref = (ss6 - n6 * f["rating_6m"] ** 2) / (n6 - 1).clip(lower=1)
    var_fut = (ss_fut - n_fut * f["future_rating"] ** 2) / (n_fut - 1).clip(lower=1)
    se = np.sqrt(var_ref.clip(lower=0.25) / n6.clip(lower=1) + var_fut.clip(lower=0.25) / n_fut.clip(lower=1))
    drop = f["ref_rating"] - f["future_rating"]
    ok = (n_fut >= MIN_REVIEWS_FUTURE) & (n6 >= MIN_REVIEWS_PAST) & f["future_rating"].notna()
    f["decline"] = np.where(ok, ((drop >= DECLINE_THRESHOLD) & (drop / se >= DECLINE_MIN_Z)).astype(float), np.nan)
    return f
