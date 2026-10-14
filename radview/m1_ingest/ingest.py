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

Usage:  python -m radview.m1_ingest.ingest [--city Philadelphia] [--max-restaurants 500]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from radview.config import PROCESSED_DIR, RAW_DIR, SOURCE, YELP_CITY, YELP_MAX_RESTAURANTS, YELP_START_DATE

BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"
CHECKIN_FILE = "yelp_academic_dataset_checkin.json"
TIP_FILE = "yelp_academic_dataset_tip.json"

_WS = re.compile(r"\s+")
_BIZ_RE = re.compile(r'"business_id":\s*"([^"]+)"')


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", text.replace("\x00", " ")).strip()


def load_restaurants(raw_dir: Path, city: str | None, max_restaurants: int | None) -> pd.DataFrame:
    """Load business.json and keep only businesses tagged as restaurants."""
    biz = pd.read_json(raw_dir / BUSINESS_FILE, lines=True)
    is_rest = biz["categories"].fillna("").str.contains(r"\bRestaurants\b", regex=True)
    biz = biz[is_rest]
    if city:
        biz = biz[biz["city"].str.lower() == city.lower()]
    if max_restaurants:
        # Prefer the restaurants with the most reviews - they make usable time series.
        biz = biz.sort_values("review_count", ascending=False).head(max_restaurants)
    cols = ["business_id", "name", "address", "city", "state", "postal_code", "latitude", "longitude",
            "stars", "review_count", "is_open", "categories"]
    biz = biz[cols].rename(columns={"stars": "yelp_stars", "review_count": "yelp_review_count"})
    biz["cuisine"] = biz["categories"].str.split(",").apply(
        lambda cs: next((c.strip() for c in cs if c.strip() not in {"Restaurants", "Food"}), "Restaurant"))
    return biz.reset_index(drop=True)


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
    rev = rev.dropna(subset=["date", "stars"])
    if start_date:
        rev = rev[rev["date"] >= pd.Timestamp(start_date)]
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
    biz = load_restaurants(raw_dir, city, max_restaurants)
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

    report = {"restaurants": len(biz), **stats, "checkin_business_months": len(ck), "tip_business_months": len(tp),
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
