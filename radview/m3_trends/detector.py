"""M3 - Emerging complaint detection.

For every restaurant and aspect we track the share of reviews that complain
about that aspect. The rate is smoothed with an EWMA (ratio of EWMA'd counts,
so sparse months do not explode) and compared with the restaurant's *own*
history using a robust z-score:

    z_t = (rate_t - median(rate_{t-W..t-1})) / scale
    scale = max(1.4826 * MAD, binomial s.e., 0.04)

The baseline window ends the month before t, so the detector only uses past
data. An aspect is flagged as a *spike* when z exceeds ``SPIKE_Z`` and there
were at least ``SPIKE_MIN_MENTIONS`` negative mentions recently.

Output: processed/aspect_trends.parquet with {aspect}_rate, {aspect}_z,
{aspect}_spike per business-month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from radview.config import ASPECTS, BASELINE_WINDOW, EWMA_SPAN, PROCESSED_DIR, SPIKE_MIN_MENTIONS, SPIKE_Z

MIN_HISTORY = 6
SCALE_FLOOR = 0.04


def _robust_z(rate: pd.Series, n_eff: pd.Series) -> pd.Series:
    past = rate.shift(1).rolling(BASELINE_WINDOW, min_periods=MIN_HISTORY)
    med = past.median()
    mad = (rate.shift(1) - med).abs().rolling(BASELINE_WINDOW, min_periods=MIN_HISTORY).median()
    binom = np.sqrt(med.clip(0.02, 0.98) * (1 - med.clip(0.02, 0.98)) / n_eff.clip(lower=1))
    scale = np.maximum.reduce([1.4826 * mad.fillna(0).to_numpy(), binom.fillna(1).to_numpy(),
                               np.full(len(rate), SCALE_FLOOR)])
    return (rate - med) / scale


def detect(panel: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    out = panel[["business_id", "month"]].copy()
    g = panel.groupby("business_id", sort=False)
    n_ewm = g["n_reviews"].transform(lambda s: s.ewm(span=EWMA_SPAN, adjust=False).mean())
    stars_ewm = g["sum_stars"].transform(lambda s: s.ewm(span=EWMA_SPAN, adjust=False).mean())
    out["n_ewm"] = n_ewm
    for a in ASPECTS:
        neg_ewm = g[f"{a}_neg"].transform(lambda s: s.ewm(span=EWMA_SPAN, adjust=False).mean())
        rate = (neg_ewm / n_ewm.where(n_ewm > 0)).fillna(0)
        out[f"{a}_rate"] = rate
        # robust z per business (groupby-apply keeps windows within a business)
        z = pd.concat([_robust_z(rate[idx], n_ewm[idx] * EWMA_SPAN) for idx in g.indices.values()]).sort_index()
        out[f"{a}_z"] = z.fillna(0).clip(-10, 10)
        recent_neg = g[f"{a}_neg"].transform(lambda s: s.rolling(2, min_periods=1).sum())
        out[f"{a}_spike"] = ((out[f"{a}_z"] > SPIKE_Z) & (recent_neg >= SPIKE_MIN_MENTIONS)).astype(np.int8)

    # Baseline "rating alarm" detector used for comparison: same robust z on the smoothed star mean.
    star_rate = (stars_ewm / n_ewm.where(n_ewm > 0))
    star_rate = star_rate.groupby(panel["business_id"]).ffill().fillna(star_rate.mean())
    zs = pd.concat([_robust_z(-star_rate[idx] / 4.0, n_ewm[idx] * EWMA_SPAN) for idx in g.indices.values()]).sort_index()
    out["rating_z"] = zs.fillna(0).clip(-10, 10)
    out["rating_alarm"] = (out["rating_z"] > SPIKE_Z).astype(np.int8)
    out["n_spiking_aspects"] = out[[f"{a}_spike" for a in ASPECTS]].sum(axis=1)

    if save:
        out.to_parquet(PROCESSED_DIR / "aspect_trends.parquet", index=False)
    print(f"[M3] flagged {int(out['n_spiking_aspects'].gt(0).sum())} business-months with >=1 complaint spike")
    return out


def evaluate(trends: pd.DataFrame, labels: pd.DataFrame, horizon_back: int = 6) -> dict:
    """How well do complaint spikes anticipate rating declines?

    * precision: P(decline in next 3 months | spike this month) vs. base rate
    * recall/lead: for each decline onset, was there a spike (or rating alarm)
      in the preceding ``horizon_back`` months, and how early was the first one?
    """
    df = trends.merge(labels[["business_id", "month", "decline"]], on=["business_id", "month"], how="inner")
    df = df.dropna(subset=["decline"])
    base = float(df["decline"].mean())
    res = {"base_rate": round(base, 4)}
    for name, col in (("complaint_spike", "n_spiking_aspects"), ("rating_alarm", "rating_alarm")):
        flagged = df[df[col] > 0]
        prec = float(flagged["decline"].mean()) if len(flagged) else float("nan")
        res[name] = {"flag_rate": round(len(flagged) / len(df), 4), "precision": round(prec, 4),
                     "lift": round(prec / base, 2) if base else None}

    # onsets: decline=1 preceded by a month with decline=0
    df = df.sort_values(["business_id", "month"])
    prev = df.groupby("business_id")["decline"].shift(1)
    onsets = df[(df["decline"] == 1) & (prev == 0)]
    pos = {(b, m): i for i, (b, m) in enumerate(zip(trends["business_id"], trends["month"]))}
    for name, col in (("complaint_spike", "n_spiking_aspects"), ("rating_alarm", "rating_alarm")):
        leads = []
        for b, m in zip(onsets["business_id"], onsets["month"]):
            i = pos[(b, m)]
            lo = max(0, i - horizon_back + 1)
            window = trends.iloc[lo:i + 1]
            window = window[window["business_id"] == b]
            hits = np.flatnonzero(window[col].to_numpy() > 0)
            # lead = months between the first alert and the start of the decline window (month after onset)
            leads.append(len(window) - hits[0] if len(hits) else np.nan)
        leads = np.array(leads, dtype=float)
        res[name]["onsets"] = int(len(leads))
        res[name]["onsets_detected_pct"] = round(float(np.mean(~np.isnan(leads))) * 100, 1) if len(leads) else None
        res[name]["median_lead_months"] = float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else None
    return res
