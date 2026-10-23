"""M3 - Emerging complaint detection.

For every restaurant and aspect we track the share of reviews that complain
about that aspect. The rate is smoothed with an EWMA (ratio of EWMA'd counts,
so sparse months do not explode) and compared with the restaurant's *own*
history using a robust z-score:

    z_t   = (rate_t - med_t) / scale_t
    med_t = median(rate_{t-W..t-1}),   MAD_t = median(|rate_{t-W..t-1} - med_t|)
    scale = max(1.4826 * MAD_t, sampling s.e., 0.04)

The baseline window ends the month before t, so the detector only uses past
data. An aspect is flagged as a *spike* when z exceeds ``SPIKE_Z`` and there
were at least ``SPIKE_MIN_MENTIONS`` negative mentions recently.

The comparison "rating alarm" applies the same test to the smoothed star
rating, with the sampling s.e. of a mean rating (star spread / sqrt(n)).

Output: processed/aspect_trends.parquet with {aspect}_rate, {aspect}_z,
{aspect}_spike per business-month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from radview.config import (ASPECTS, BASELINE_WINDOW, DECLINE_THRESHOLD, EWMA_SPAN, PROCESSED_DIR, RANDOM_SEED,
                            SPIKE_MIN_MENTIONS, SPIKE_Z)

MIN_HISTORY = 6
SCALE_FLOOR = 0.04
MIN_STAR_VAR = 0.25   # floor on the star variance, as in the M4 significance test
N_SHUFFLES = 20       # random-timing replicates for the chance baseline


def _baseline(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD of the previous BASELINE_WINDOW values, for one business (NaN until MIN_HISTORY)."""
    padded = np.concatenate([np.full(BASELINE_WINDOW, np.nan), values.astype(float)])
    win = sliding_window_view(padded[:-1], BASELINE_WINDOW)          # row t = values[t-W .. t-1]
    enough = (~np.isnan(win)).sum(axis=1) >= MIN_HISTORY
    med = np.full(len(values), np.nan)
    mad = np.full(len(values), np.nan)
    if enough.any():
        w = win[enough]
        med[enough] = np.nanmedian(w, axis=1)
        mad[enough] = np.nanmedian(np.abs(w - med[enough][:, None]), axis=1)
    return med, mad


def _robust_z(values: pd.Series, groups: dict[str, np.ndarray], se_of_median) -> pd.Series:
    """Robust z within each business (``groups`` = row positions per business, windows never cross them).

    ``se_of_median(med, idx)`` gives the sampling s.e. for those rows given their baseline median.
    """
    v_all = values.to_numpy(dtype=float)
    z = np.full(len(v_all), np.nan)
    for idx in groups.values():
        v = v_all[idx]
        med, mad = _baseline(v)
        se = np.nan_to_num(se_of_median(med, idx), nan=1.0)
        scale = np.maximum.reduce([1.4826 * np.nan_to_num(mad), se, np.full(len(v), SCALE_FLOOR)])
        z[idx] = (v - med) / scale
    return pd.Series(z, index=values.index)


def detect(panel: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    panel = panel.reset_index(drop=True)
    out = panel[["business_id", "month"]].copy()
    g = panel.groupby("business_id", sort=False)

    def ewm(col):
        return g[col].transform(lambda s: s.ewm(span=EWMA_SPAN, adjust=False).mean())

    n_ewm = ewm("n_reviews")
    n_eff = (n_ewm * EWMA_SPAN).clip(lower=1).to_numpy()  # effective review count behind a span-3 EWMA ratio
    out["n_ewm"] = n_ewm
    for a in ASPECTS:
        rate = (ewm(f"{a}_neg") / n_ewm.where(n_ewm > 0)).fillna(0)
        out[f"{a}_rate"] = rate

        def binom_se(med, idx):
            p = np.clip(med, 0.02, 0.98)
            return np.sqrt(p * (1 - p) / n_eff[idx])
        out[f"{a}_z"] = _robust_z(rate, g.indices, binom_se).fillna(0).clip(-10, 10)
        recent_neg = g[f"{a}_neg"].transform(lambda s: s.rolling(2, min_periods=1).sum())
        out[f"{a}_spike"] = ((out[f"{a}_z"] > SPIKE_Z) & (recent_neg >= SPIKE_MIN_MENTIONS)).astype(np.int8)

    # Baseline "rating alarm" detector used for comparison: same robust z on the smoothed star mean
    # (scaled by 1/4 so the 0.04 floor means 0.16 stars), with the s.e. of a mean rating.
    mean_stars = (ewm("sum_stars") / n_ewm.where(n_ewm > 0))
    if "sumsq_stars" in panel:
        var_stars = (ewm("sumsq_stars") / n_ewm.where(n_ewm > 0) - mean_stars ** 2).clip(lower=MIN_STAR_VAR)
    else:  # hand-built panels (tests) may omit it: assume a typical spread of one star
        var_stars = pd.Series(1.0, index=panel.index)
    filled = mean_stars.groupby(panel["business_id"]).ffill().fillna(mean_stars.mean())
    star_se = (np.sqrt(var_stars.groupby(panel["business_id"]).ffill().fillna(1.0).to_numpy() / n_eff) / 4.0)
    out["rating_z"] = _robust_z(-filled / 4.0, g.indices, lambda med, idx: star_se[idx]).fillna(0).clip(-10, 10)
    out["rating_alarm"] = (out["rating_z"] > SPIKE_Z).astype(np.int8)
    out["n_spiking_aspects"] = out[[f"{a}_spike" for a in ASPECTS]].sum(axis=1)

    if save:
        out.to_parquet(PROCESSED_DIR / "aspect_trends.parquet", index=False)
    print(f"[M3] flagged {int(out['n_spiking_aspects'].gt(0).sum())} business-months with >=1 complaint spike")
    return out


def _lead_times(flag: np.ndarray, onset: np.ndarray, biz: np.ndarray, horizon_back: int) -> np.ndarray:
    """Months from the first alert in the ``horizon_back`` months up to each onset to the start of its decline
    window (the month after the onset); NaN when no alert fired. Rows are consecutive calendar months."""
    leads = []
    for i in np.flatnonzero(onset):
        lo = max(0, i - horizon_back + 1)
        while biz[lo] != biz[i]:
            lo += 1
        hits = np.flatnonzero(flag[lo:i + 1])
        leads.append(i + 1 - (lo + hits[0]) if len(hits) else np.nan)
    return np.array(leads, dtype=float)


def _shuffle_within_business(flag: np.ndarray, biz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same alerts per restaurant, at random times (circular shift keeps their clustering)."""
    out = flag.copy()
    starts = np.flatnonzero(np.r_[True, biz[1:] != biz[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(biz)]):
        if e - s > 1:
            out[s:e] = np.roll(flag[s:e], int(rng.integers(1, e - s)))
    return out


def evaluate(trends: pd.DataFrame, labels: pd.DataFrame, horizon_back: int = 6) -> dict:
    """How well do complaint spikes anticipate rating declines?

    * precision: P(decline in next 3 months | spike this month) vs. base rate
    * recall/lead: for each decline onset (labelled 0 the month before, 1 now), was there a spike (or rating
      alarm) in the preceding ``horizon_back`` months, and how early was the first one?
    * lasting_drop: P(the next 3 months are >= DECLINE_THRESHOLD below the 6-month rating from *before* the
      alert, i.e. 3 months earlier). The decline label compares the future with a window that already contains
      the alert month, so a detector of damage that is already visible scores poorly on it (regression to the
      mean); this target asks instead whether the alert marks damage that lasts.
    * chance: the same metrics with each restaurant's alerts moved to random months - the level a detector
      with no timing information reaches.
    """
    cols = ["business_id", "month", "decline"] + [c for c in ("rating_6m", "future_rating") if c in labels]
    df = (trends.merge(labels[cols], on=["business_id", "month"], how="inner")
          .sort_values(["business_id", "month"]).reset_index(drop=True))
    biz = df["business_id"].to_numpy()
    y = df["decline"].to_numpy()
    labelled = ~np.isnan(y)
    prev = df.groupby("business_id")["decline"].shift(1).to_numpy()   # the calendar month before
    onset = (y == 1) & (prev == 0)
    base = float(y[labelled].mean())
    lasting = None
    if "rating_6m" in df and "future_rating" in df:
        ref_pre = df.groupby("business_id")["rating_6m"].shift(3).to_numpy()
        ok = labelled & ~np.isnan(ref_pre)
        lasting = (ok, (df["future_rating"].to_numpy() <= ref_pre - DECLINE_THRESHOLD).astype(float))

    def score(flag: np.ndarray) -> dict:
        f = flag[labelled]
        prec = float(y[labelled][f].mean()) if f.any() else float("nan")
        leads = _lead_times(flag, onset, biz, horizon_back)
        r = {"flag_rate": round(float(f.mean()), 4), "precision": round(prec, 4),
             "lift": round(prec / base, 2) if base else None, "onsets": int(len(leads)),
             "onsets_detected_pct": round(float(np.mean(~np.isnan(leads))) * 100, 1) if len(leads) else None,
             "median_lead_months": float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else None}
        if lasting is not None:
            ok, ev = lasting
            b, hit = float(ev[ok].mean()), ok & flag
            p = float(ev[hit].mean()) if hit.any() else float("nan")
            r["lasting_drop"] = {"base_rate": round(b, 4), "precision": round(p, 4),
                                 "lift": round(p / b, 2) if b else None}
        return r

    def mean(values, digits):  # None (e.g. no alerts at all) counts as missing
        vals = np.array([np.nan if v is None else v for v in values], dtype=float)
        return round(float(np.nanmean(vals)), digits) if (~np.isnan(vals)).any() else None

    res = {"base_rate": round(base, 4)}
    rng = np.random.default_rng(RANDOM_SEED)
    for name, col in (("complaint_spike", "n_spiking_aspects"), ("rating_alarm", "rating_alarm")):
        flag = df[col].to_numpy() > 0
        res[name] = score(flag)
        reps = [score(_shuffle_within_business(flag, biz, rng)) for _ in range(N_SHUFFLES)]
        chance = {"lift": mean([r["lift"] for r in reps], 2),
                  "onsets_detected_pct": mean([r["onsets_detected_pct"] for r in reps], 1)}
        if lasting is not None:
            chance["lasting_drop_lift"] = mean([r["lasting_drop"]["lift"] for r in reps], 2)
        res[name]["chance"] = chance
    return res
