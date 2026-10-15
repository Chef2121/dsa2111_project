"""End-to-end Radview pipeline (M1 -> M5).

Modes
    full     ingest, train + evaluate every model, score, generate plans
             (initial Kubernetes Job / local setup)
    rescore  ingest new data, re-score with the saved models, regenerate plans
             (weekly Kubernetes CronJob - no retraining)

The data source is chosen by RADVIEW_SOURCE (``yelp`` or ``synthetic``; default:
``yelp`` when data/yelp/ holds the Yelp files). Synthetic data is generated
automatically if it does not exist yet.

Usage:
    python -m radview.pipeline                      # full run, auto source
    python -m radview.pipeline --fast               # skip DistilBERT (TF-IDF only)
    python -m radview.pipeline --mode rescore
    RADVIEW_SOURCE=synthetic python -m radview.pipeline
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from radview.config import (OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR, SOURCE, YELP_CITY, YELP_MAX_RESTAURANTS,
                            YELP_START_DATE)


def _step(name: str):
    print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
    return time.time()


def build_trends_and_features() -> dict:
    """Panel (shared) -> M3 complaint trends -> M4 features; returns the M3 evaluation."""
    from radview.m3_trends.detector import detect, evaluate as evaluate_m3
    from radview.m4_forecast.features import build_features
    from radview.panel import build_panel

    panel = build_panel()
    trends = detect(panel)
    import pandas as pd

    cities = pd.read_parquet(PROCESSED_DIR / "businesses.parquet").set_index("business_id")["city"]
    feat = build_features(panel, trends, cities)
    feat.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    m3 = evaluate_m3(trends, feat)
    (OUTPUTS_DIR / "m3_evaluation.json").write_text(json.dumps(m3, indent=2))
    print("[M3]", json.dumps(m3))
    return m3


def run(mode: str = "full", fast: bool = False, use_llm: bool | None = None, epochs: int = 2,
        train_size: int = 6000) -> dict:
    timings = {}
    t = _step(f"M1 ingestion ({SOURCE})")
    if SOURCE == "synthetic" and not (RAW_DIR / "yelp_academic_dataset_review.json").exists():
        from radview.m1_ingest.synthetic import generate
        generate(out_dir=RAW_DIR)
    from radview.m1_ingest.ingest import run as ingest
    city, cap, start = (YELP_CITY, YELP_MAX_RESTAURANTS, YELP_START_DATE) if SOURCE == "yelp" else (None, None, None)
    ingest(city=city, max_restaurants=cap, start_date=start)
    timings["m1"] = time.time() - t

    t = _step("M2 aspect classification")
    from radview.m2_aspects import train as m2
    if mode == "full":
        m2.run(use_transformer=not fast, epochs=epochs, train_size=train_size)
    else:
        m2.score_all()
    timings["m2"] = time.time() - t

    t = _step("M3 trends + M4 features")
    build_trends_and_features()
    timings["m3"] = time.time() - t

    t = _step("M4 decline forecasting")
    from radview.m4_forecast import train as m4
    if mode == "full":
        m4.run()
    else:
        m4.score_saved()
    timings["m4"] = time.time() - t

    t = _step("M5 agentic action plans")
    from radview.m5_agent.agent import run_batch
    from radview.m5_agent.cli import compare
    m5 = run_batch(use_llm=use_llm)
    compare(20)  # rule-based vs adversarial (vs Claude when a key is set)
    timings["m5"] = time.time() - t

    summary = {"source": SOURCE, "mode": mode, "finished_at": datetime.now().isoformat(timespec="seconds"),
               "timings_s": {k: round(v, 1) for k, v in timings.items()}, "m5": m5}
    (OUTPUTS_DIR / "pipeline_run.json").write_text(json.dumps(summary, indent=2))
    print("\n[pipeline] done:", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "rescore"], default="full")
    ap.add_argument("--fast", action="store_true", help="skip transformer fine-tuning")
    ap.add_argument("--epochs", type=int, default=2, help="transformer fine-tuning epochs")
    ap.add_argument("--train-size", type=int, default=6000, help="weakly-labelled reviews for transformer fine-tuning")
    ap.add_argument("--llm", dest="use_llm", action="store_true", default=None, help="force LLM drafting in M5")
    ap.add_argument("--no-llm", dest="use_llm", action="store_false", help="force deterministic M5 planner")
    args = ap.parse_args()
    run(mode=args.mode, fast=args.fast, use_llm=args.use_llm, epochs=args.epochs, train_size=args.train_size)
