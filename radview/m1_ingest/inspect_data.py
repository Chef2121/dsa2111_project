"""Dataset preflight: look before you ingest.

Reads only business.json (~120 MB, seconds) and prints, per city, how many
restaurants there are and how many reviews they carry, plus the command line
for a sensible slice. One city is the right unit of analysis: market-relative
features compare a restaurant with the restaurants it actually competes with.

Usage:  python -m radview.m1_ingest.inspect_data [--top 15]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from radview.config import RAW_DIR
from radview.m1_ingest.ingest import BUSINESS_FILE


def summarise(raw_dir=RAW_DIR, top: int = 15) -> pd.DataFrame:
    rows = []
    with open(raw_dir / BUSINESS_FILE, encoding="utf-8") as fh:
        for line in fh:
            b = json.loads(line)
            if b.get("categories") and "Restaurants" in b["categories"]:
                rows.append((b["city"], b["state"], b["review_count"], b["is_open"]))
    df = pd.DataFrame(rows, columns=["city", "state", "review_count", "is_open"])
    by_city = (df.groupby(["city", "state"])
                 .agg(restaurants=("review_count", "size"), reviews=("review_count", "sum"),
                      open_share=("is_open", "mean"), median_reviews=("review_count", "median"))
                 .sort_values("reviews", ascending=False).head(top).reset_index())
    by_city["open_share"] = by_city["open_share"].round(2)
    return by_city


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    table = summarise(top=args.top)
    print(table.to_string(index=False))
    best = table.iloc[0]
    print(f"\nSuggested: RADVIEW_CITY=\"{best['city']}\" RADVIEW_MAX_RESTAURANTS=400 python -m radview.pipeline"
          f"\n(top 400 most-reviewed restaurants of {int(best['restaurants'])} in {best['city']}; "
          f"set RADVIEW_START_DATE to trade history for speed)")
