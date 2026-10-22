"""M4 - second opinion on *why* a restaurant is at risk.

SHAP (in ``model.explain``) answers "which inputs moved this prediction". Two
complementary, owner-readable quantities are computed here:

* **Counterfactual attribution** - ask the trained GRU: what would this
  restaurant's risk be if one aspect looked like the *typical restaurant in its
  market* (complaint rate at the market median, no growth, no spike) for the
  whole 12-month window? The drop in risk is that aspect's attributed risk.
  It is an intervention on the model, so it captures interactions SHAP's
  additive split can blur - but it is associational, not causal.

* **Star cost** - a regression of monthly mean stars on per-aspect complaint
  rates across the market, with every coefficient constrained to be <= 0 (a
  complaint can cost stars, never earn them). Its coefficients read as "every extra 10%
  of reviews complaining about service costs about X stars"; multiplied by how
  far a restaurant sits above the market median, it estimates how many stars
  that problem is costing it today.

Reporting SHAP and the counterfactual side by side (and how often they agree on
the top driver) is itself an evaluation result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from radview.config import ASPECTS

MIN_REVIEWS_FOR_FIT = 3


def fit_star_impact(panel: pd.DataFrame) -> dict[str, float]:
    """Stars lost per unit complaint rate for each aspect (negative numbers = cost)."""
    df = panel[panel["n_reviews"] >= MIN_REVIEWS_FOR_FIT]
    X = np.column_stack([df[f"{a}_neg"] / df["n_reviews"] for a in ASPECTS])
    # Aspect rates are correlated, so unconstrained coefficients can flip sign (on Yelp, ambience came out
    # +0.37: more complaints, more stars), which makes an owner-facing explanation nonsensical. Fitting on the
    # negated rates with positive=True forces every effect to be a cost. Months are weighted by review count,
    # normalised to mean 1 so that alpha keeps its usual scale (raw counts made the penalty negligible).
    w = df["n_reviews"] / df["n_reviews"].mean()
    model = Ridge(alpha=1.0, positive=True).fit(-X, df["mean_stars"], sample_weight=w)
    return {a: round(float(-c), 4) + 0.0 for a, c in zip(ASPECTS, model.coef_)}  # + 0.0 turns -0.0 into 0.0


def typical_aspect_frame(feat: pd.DataFrame, aspect: str) -> pd.DataFrame:
    """Copy of ``feat`` where ``aspect`` looks like the market's typical restaurant in every month."""
    cf = feat.copy()
    cf[f"{aspect}_neg_rate_3m"] = cf[f"{aspect}_market_rate"].fillna(cf[f"{aspect}_neg_rate_3m"])
    for suffix in ("neg_rate_change", "neg_rate_vs_market", "z", "spike_3m"):
        cf[f"{aspect}_{suffix}"] = 0.0
    if f"{aspect}_spike_3m" in cf:
        cf["n_spiking_aspects"] = (cf["n_spiking_aspects"] - feat[f"{aspect}_spike_3m"]).clip(lower=0)
    return cf


def counterfactual_attribution(feat: pd.DataFrame, rows: np.ndarray, risk: np.ndarray, predict_fn) -> pd.DataFrame:
    """Risk reduction if each aspect were at the market norm. ``predict_fn(frame, rows) -> risk``."""
    out = {}
    for a in ASPECTS:
        cf_risk = predict_fn(typical_aspect_frame(feat, a), rows)
        out[a] = risk - cf_risk
    return pd.DataFrame(out)


def star_costs(latest: pd.DataFrame, coefs: dict[str, float]) -> pd.DataFrame:
    """Estimated stars lost vs. the market, per aspect (0 when at or below the market median)."""
    return pd.DataFrame({
        a: (latest[f"{a}_neg_rate_3m"] - latest[f"{a}_market_rate"]).clip(lower=0).fillna(0) * max(-coefs[a], 0.0)
        for a in ASPECTS})


def method_agreement(shap_by_aspect: pd.DataFrame, cf_by_aspect: pd.DataFrame, mask: np.ndarray) -> dict:
    """How often SHAP and the counterfactual pick the same top aspect (and rank correlation)."""
    if mask.sum() == 0:
        return {"n": 0}
    s, c = shap_by_aspect[mask], cf_by_aspect[mask]
    top_same = (s.idxmax(axis=1) == c.idxmax(axis=1)).mean()
    rho = [pd.Series(s.iloc[i].to_numpy()).corr(pd.Series(c.iloc[i].to_numpy()), method="spearman")
           for i in range(len(s))]
    return {"n": int(mask.sum()), "top_aspect_agreement": round(float(top_same), 4),
            "mean_spearman": round(float(np.nanmean(rho)), 4)}
