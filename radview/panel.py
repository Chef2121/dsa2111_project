"""Monthly business-level panel shared by M3 and M4.

Turns review-level data (M1) and aspect predictions (M2) into one row per
restaurant per calendar month, including months with zero reviews so that
rolling windows are true calendar windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from radview.config import ASPECTS, LABEL_NEG, LABEL_POS, PROCESSED_DIR


def build_panel(processed_dir=PROCESSED_DIR, save: bool = True) -> pd.DataFrame:
    rev = pd.read_parquet(processed_dir / "reviews.parquet")
    asp = pd.read_parquet(processed_dir / "review_aspects.parquet")
    df = rev.merge(asp, on="review_id", how="inner")

    df["stars_sq"] = df["stars"] ** 2
    agg = {"n_reviews": ("review_id", "size"), "sum_stars": ("stars", "sum"), "sumsq_stars": ("stars_sq", "sum"),
           "n_low_star": ("stars", lambda s: int((s <= 2).sum()))}
    for a in ASPECTS:
        df[f"{a}_neg"] = (df[f"{a}_label"] == LABEL_NEG).astype(int)
        df[f"{a}_pos"] = (df[f"{a}_label"] == LABEL_POS).astype(int)
        agg[f"{a}_neg"] = (f"{a}_neg", "sum")
        agg[f"{a}_pos"] = (f"{a}_pos", "sum")
    df["any_neg"] = df[[f"{a}_neg" for a in ASPECTS]].max(axis=1)
    agg["any_neg"] = ("any_neg", "sum")
    monthly = df.groupby(["business_id", "month"]).agg(**agg).reset_index()

    # Full calendar grid: first review month of each business -> last month in data.
    last = pd.Period(monthly["month"].max(), "M")
    firsts = monthly.groupby("business_id")["month"].min()
    grid = pd.concat([
        pd.DataFrame({"business_id": b, "month": pd.period_range(pd.Period(m, "M"), last, freq="M").astype(str)})
        for b, m in firsts.items()
    ], ignore_index=True)
    panel = grid.merge(monthly, on=["business_id", "month"], how="left").fillna(0)

    for extra, col in (("checkins_monthly.parquet", "checkins"), ("tips_monthly.parquet", "tips")):
        path = processed_dir / extra
        if path.exists():
            panel = panel.merge(pd.read_parquet(path), on=["business_id", "month"], how="left")
        panel[col] = panel.get(col, 0)
        panel[col] = panel[col].fillna(0)

    int_cols = [c for c in panel.columns if c not in {"business_id", "month", "sum_stars", "sumsq_stars"}]
    panel[int_cols] = panel[int_cols].astype(np.int64)
    panel["mean_stars"] = np.where(panel["n_reviews"] > 0, panel["sum_stars"] / panel["n_reviews"].clip(lower=1), np.nan)
    panel = panel.sort_values(["business_id", "month"]).reset_index(drop=True)
    if save:
        panel.to_parquet(processed_dir / "panel_monthly.parquet", index=False)
    return panel
