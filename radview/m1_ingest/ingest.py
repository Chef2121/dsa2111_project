"""M1 - Data ingestion & storage.

Reads the Yelp Open Dataset JSON-lines files (real or synthetic), keeps only
restaurants, cleans and de-duplicates reviews, and writes columnar parquet
tables that every downstream module consumes:

    processed/businesses.parquet   one row per restaurant
    processed/reviews.parquet      cleaned reviews (review_id, business_id, stars, text, date, month)
    processed/checkins_monthly.parquet   check-in counts per business-month
    processed/tips_monthly.parquet       tip counts per business-month
    processed/ingest_report.json   data-quality report

Large files are streamed in chunks so the real 5 GB review file never has to
fit in memory. ``--city`` / ``--max-restaurants`` let you carve out a slice of
the real dataset that is tractable on a laptop.

Which restaurants make the slice (``RADVIEW_SELECT_BY``):
    lifetime  the N with the highest Yelp ``review_count`` (default). That count is
              taken at the end of the dataset, so the slice favours restaurants that
              survived and stayed popular - a selection that uses future information.
    early     the N with the most reviews in the first 12 months after the start
              date: only information available at the start, so restaurants that
              later declined or closed are kept in proportion.

Usage:  python -m radview.m1_ingest.ingest [--city Philadelphia] [--max-restaurants 500]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from radview.config import (PROCESSED_DIR, RAW_DIR, SELECT_BY, SOURCE, YELP_CITY, YELP_MAX_RESTAURANTS,
                            YELP_START_DATE)

BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"
CHECKIN_FILE = "yelp_academic_dataset_checkin.json"
TIP_FILE = "yelp_academic_dataset_tip.json"
EARLY_WINDOW_MONTHS = 12

_WS = re.compile(r"\s+")
_BIZ_RE = re.compile(r'"business_id":\s*"([^"]+)"')
_DATE_RE = re.compile(r'"date":\s*"(\d{4}-\d{2}-\d{2})')

# Yelp lists categories in no meaningful order, so "first category" is often a venue or drinks tag
# ("Nightlife", "Bars"). These tags never describe the kind of food and are skipped as a cuisine.
NON_CUISINE = {
    "Restaurants", "Food", "Nightlife", "Bars", "Cocktail Bars", "Beer", "Wine & Spirits", "Pubs", "Lounges",
    "Wine Bars", "Beer Bar", "Sports Bars", "Dive Bars", "Whiskey Bars", "Hookah Bars", "Beer Gardens", "Breweries",
    "Dance Clubs", "Music Venues", "Jazz & Blues", "Karaoke", "Arts & Entertainment", "Active Life", "Shopping",
    "Event Planning & Services", "Venues & Event Spaces", "Party & Event Planning", "Caterers",
    "Food Delivery Services", "Food Trucks", "Food Stands", "Local Flavor", "Specialty Food", "Ethnic Food",
    "Coffee & Tea", "Juice Bars & Smoothies", "Desserts", "Ice Cream & Frozen Yogurt", "Chocolatiers & Shops",
    "Cheese Shops", "Fruits & Veggies", "Grocery", "Convenience Stores", "Imported Food", "Vegetarian", "Vegan",
    "Gluten-Free", "Halal", "Kosher", "Hotels & Travel", "Hotels",
}


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", text.replace("\x00", " ")).strip()


def assign_cuisine(categories: pd.Series) -> pd.Series:
    """One cuisine per restaurant, used to pick comparable peers.

    Of each restaurant's food categories (venue, drinks and diet tags skipped), the one most common across the
    whole slice wins, so peer groups are as large as possible; ties keep Yelp's listed order.
    """
    lists = categories.fillna("").map(lambda s: [c.strip() for c in s.split(",") if c.strip() not in NON_CUISINE])
    counts = Counter(c for cs in lists for c in cs)
    return lists.map(lambda cs: max(cs, key=lambda c: (counts[c], -cs.index(c))) if cs else "Restaurant")


def early_review_counts(raw_dir: Path, business_ids: set[str], start_date: str,
                        months: int = EARLY_WINDOW_MONTHS) -> Counter:
    """Reviews per business in the first ``months`` months from ``start_date`` (one cheap streaming pass)."""
    start = pd.Timestamp(start_date)
    lo, hi = str(start.date()), str((start + pd.DateOffset(months=months)).date())
    counts: Counter = Counter()
    with open(raw_dir / REVIEW_FILE, encoding="utf-8") as fh:
        for line in fh:
            m = _BIZ_RE.search(line)
            if not m or m.group(1) not in business_ids:
                continue
            d = _DATE_RE.search(line)
            if d and lo <= d.group(1) < hi:
                counts[m.group(1)] += 1
    return counts


def load_restaurants(raw_dir: Path, city: str | None, max_restaurants: int | None,
                     select_by: str = SELECT_BY, start_date: str | None = None) -> pd.DataFrame:
    """Load business.json and keep only businesses tagged as restaurants."""
    biz = pd.read_json(raw_dir / BUSINESS_FILE, lines=True)
    is_rest = biz["categories"].fillna("").str.contains(r"\bRestaurants\b", regex=True)
    biz = biz[is_rest]
    if city:
        biz = biz[biz["city"].str.lower() == city.lower()]
    if max_restaurants:
        # Prefer the restaurants with the most reviews - they make usable time series.
        # business_id breaks ties at the cut-off (3 restaurants share the 400th place on Philadelphia), so the
        # same files always give the same slice
        if select_by == "early" and start_date:
            counts = early_review_counts(raw_dir, set(biz["business_id"]), start_date)
            biz = biz.assign(_early=biz["business_id"].map(counts).fillna(0))
            biz = biz.sort_values(["_early", "review_count", "business_id"], ascending=[False, False, True])
        else:
            biz = biz.sort_values(["review_count", "business_id"], ascending=[False, True])
        biz = biz.head(max_restaurants)
    cols = ["business_id", "name", "address", "city", "state", "postal_code", "latitude", "longitude",
            "stars", "review_count", "is_open", "categories"]
    biz = biz[cols].rename(columns={"stars": "yelp_stars", "review_count": "yelp_review_count"})
    biz["cuisine"] = assign_cuisine(biz["categories"])
    return biz.reset_index(drop=True)


def refresh_cuisine(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Recompute the cuisine column of an existing businesses.parquet (no raw data needed)."""
    path = out_dir / "businesses.parquet"
    biz = pd.read_parquet(path)
    biz["cuisine"] = assign_cuisine(biz["categories"])
    biz.to_parquet(path, index=False)
    return biz


def load_reviews(raw_dir: Path, business_ids: set[str], start_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Stream review.json line by line, keep restaurant reviews, clean and de-duplicate.

    The real file is ~5 GB / 7M lines, so we never load it whole: each line is
    parsed and kept only if it belongs to a selected restaurant.
    """
    stats = {"rows_read": 0, "rows_kept_restaurants": 0}
    keep_cols = ("review_id", "user_id", "business_id", "stars", "useful", "text", "date")
    rows = []
    with open(raw_dir / REVIEW_FILE, encoding="utf-8") as fh:
        for line in fh:
            stats["rows_read"] += 1
            # cheap pre-filter on the raw line before paying for json.loads
            m = _BIZ_RE.search(line)
            if not m or m.group(1) not in business_ids:
                continue
            rec = json.loads(line)
            rows.append({k: rec.get(k) for k in keep_cols})
    stats["rows_kept_restaurants"] = len(rows)
    rev = pd.DataFrame(rows, columns=list(keep_cols))

    rev["text"] = rev["text"].map(clean_text)
    rev["date"] = pd.to_datetime(rev["date"], errors="coerce")
    n0 = len(rev)
    if start_date:
        rev = rev[rev["date"].isna() | (rev["date"] >= pd.Timestamp(start_date))]
    stats["dropped_before_start_date"] = n0 - len(rev)
    n0 = len(rev)
    rev = rev.dropna(subset=["date", "stars"])
    rev = rev[rev["text"].str.len() >= 10]
    stats["dropped_invalid_or_empty"] = n0 - len(rev)

    n1 = len(rev)
    rev = rev.drop_duplicates(subset="review_id")
    rev = rev.drop_duplicates(subset=["user_id", "business_id", "text"])
    stats["dropped_duplicates"] = n1 - len(rev)

    rev["stars"] = rev["stars"].astype(float).clip(1, 5)
    # A dataset snapshot usually ends part-way through a month; that stub month would look
    # like a collapse in review volume, so drop it.
    last = rev["date"].max()
    if last.day < 25:
        n2 = len(rev)
        rev = rev[rev["date"] < last.to_period("M").start_time]
        stats["dropped_partial_last_month"] = n2 - len(rev)
    rev["month"] = rev["date"].dt.to_period("M").astype(str)
    rev = rev.sort_values(["business_id", "date"]).reset_index(drop=True)
    stats["reviews_final"] = len(rev)
    return rev, stats


def load_checkins(raw_dir: Path, business_ids: set[str]) -> pd.DataFrame:
    """Explode the comma-separated check-in timestamps into business-month counts."""
    path = raw_dir / CHECKIN_FILE
    if not path.exists():
        return pd.DataFrame(columns=["business_id", "month", "checkins"])
    ck = pd.read_json(path, lines=True)
    ck = ck[ck["business_id"].isin(business_ids)]
    ck = ck.assign(date=ck["date"].str.split(", ")).explode("date")
    ck["month"] = pd.to_datetime(ck["date"], errors="coerce").dt.to_period("M").astype(str)
    return ck.groupby(["business_id", "month"]).size().rename("checkins").reset_index()


def load_tips(raw_dir: Path, business_ids: set[str]) -> pd.DataFrame:
    path = raw_dir / TIP_FILE
    if not path.exists():
        return pd.DataFrame(columns=["business_id", "month", "tips"])
    tp = pd.read_json(path, lines=True)
    tp = tp[tp["business_id"].isin(business_ids)]
    tp["month"] = pd.to_datetime(tp["date"], errors="coerce").dt.to_period("M").astype(str)
    return tp.groupby(["business_id", "month"]).size().rename("tips").reset_index()


def run(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, city: str | None = None,
        max_restaurants: int | None = None, start_date: str | None = None) -> dict:
    if not (raw_dir / BUSINESS_FILE).exists():
        raise FileNotFoundError(
            f"No Yelp data in {raw_dir}. Place the Yelp Open Dataset JSON files there or run "
            "`python -m radview.m1_ingest.synthetic` to generate a synthetic sample.")
    biz = load_restaurants(raw_dir, city, max_restaurants, start_date=start_date)
    ids = set(biz["business_id"])
    rev, stats = load_reviews(raw_dir, ids, start_date)
    biz = biz[biz["business_id"].isin(set(rev["business_id"]))].reset_index(drop=True)
    ck = load_checkins(raw_dir, ids)
    tp = load_tips(raw_dir, ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    biz.to_parquet(out_dir / "businesses.parquet", index=False)
    rev.to_parquet(out_dir / "reviews.parquet", index=False)
    ck.to_parquet(out_dir / "checkins_monthly.parquet", index=False)
    tp.to_parquet(out_dir / "tips_monthly.parquet", index=False)

    report = {"restaurants": len(biz), "selected_by": SELECT_BY if max_restaurants else "all", **stats,
              "checkin_business_months": len(ck), "tip_business_months": len(tp),
              "date_min": str(rev["date"].min()), "date_max": str(rev["date"].max()),
              "mean_stars": round(float(rev["stars"].mean()), 3)}
    (out_dir / "ingest_report.json").write_text(json.dumps(report, indent=2))
    print("[M1 ingest]", report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=YELP_CITY if SOURCE == "yelp" else None)
    ap.add_argument("--max-restaurants", type=int, default=YELP_MAX_RESTAURANTS if SOURCE == "yelp" else None)
    ap.add_argument("--start-date", default=YELP_START_DATE if SOURCE == "yelp" else None)
    args = ap.parse_args()
    run(city=args.city, max_restaurants=args.max_restaurants, start_date=args.start_date)
