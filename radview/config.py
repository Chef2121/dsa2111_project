"""Central configuration for Radview.

Every path and tunable used by the pipeline lives here so that the same code
runs unchanged on a laptop, in Docker, or inside a Kubernetes pod. Paths can be
overridden with environment variables (see README / k8s ConfigMap).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RADVIEW_DATA_DIR", ROOT / "data"))

# Two data sources share one code path:
#   "yelp"      - the real Yelp Open Dataset JSON files in data/yelp/
#   "synthetic" - the Yelp-format simulator output in data/synthetic/raw/
# Each source gets its own working directory so results never overwrite each other.
_YELP_DIR = DATA_DIR / "yelp"
# Yelp is chosen when its raw files are present *or* a finished Yelp run is (e.g. teammates who only
# received the trained-models bundle, not the 5 GB raw dataset).
SOURCE = os.getenv("RADVIEW_SOURCE") or (
    "yelp" if (_YELP_DIR / "yelp_academic_dataset_review.json").exists()
    or (DATA_DIR / "work" / "yelp" / "outputs" / "risk_latest.parquet").exists() else "synthetic")
RAW_DIR = Path(os.getenv("RADVIEW_RAW_DIR", _YELP_DIR if SOURCE == "yelp" else DATA_DIR / "synthetic" / "raw"))
WORK_DIR = DATA_DIR / "work" / SOURCE
PROCESSED_DIR = WORK_DIR / "processed"  # cleaned parquet tables, panel, trends
MODELS_DIR = WORK_DIR / "models"        # trained M2 / M4 artefacts
OUTPUTS_DIR = WORK_DIR / "outputs"      # scores, plans, evaluation reports
LABELS_DIR = DATA_DIR / "labels" / SOURCE  # hand-labelled aspect test set
SEMEVAL_FILE = DATA_DIR / "semeval" / "Restaurants_Test_Gold.xml"  # human-labelled ABSA benchmark
SEMEVAL_TRAIN_FILE = DATA_DIR / "semeval" / "Restaurants_Train_v2.xml"  # source of the model-selection dev set

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, LABELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Slice of the real dataset used by default (CPU-friendly; override via env).
YELP_CITY = os.getenv("RADVIEW_CITY", "Philadelphia")
YELP_MAX_RESTAURANTS = int(os.getenv("RADVIEW_MAX_RESTAURANTS", "400"))
YELP_START_DATE = os.getenv("RADVIEW_START_DATE", "2016-01-01")  # 6 years of history keeps CPU scoring manageable
# How the N restaurants are chosen: "lifetime" Yelp review count (uses future information, favours survivors)
# or "early" = most reviews in the first 12 months after the start date (see m1_ingest/ingest.py).
SELECT_BY = os.getenv("RADVIEW_SELECT_BY", "lifetime")

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------
ASPECTS = ["food", "service", "cleanliness", "price", "ambience"]

# Per-aspect label encoding used by every M2 model.
LABEL_NONE, LABEL_POS, LABEL_NEG = 0, 1, 2
LABEL_NAMES = {LABEL_NONE: "not mentioned", LABEL_POS: "positive", LABEL_NEG: "negative"}

# --------------------------------------------------------------------------
# M3 - emerging complaint detection
# --------------------------------------------------------------------------
EWMA_SPAN = 3            # months; smoothing of the monthly complaint rate
BASELINE_WINDOW = 12     # months of history used for the robust baseline
SPIKE_Z = 2.5            # robust z-score above which an aspect is flagged
SPIKE_MIN_MENTIONS = 2   # need at least this many negative mentions in the last two months

# --------------------------------------------------------------------------
# M4 - rating decline forecasting
# --------------------------------------------------------------------------
HORIZON_MONTHS = 3          # look-ahead window
DECLINE_THRESHOLD = 0.3     # stars: future mean must fall at least this much ...
DECLINE_MIN_Z = 1.645       # ... and the drop must be significant (one-sided z-test, p < 0.05)
REFERENCE_MONTHS = 6        # trailing window that defines the "current" rating
MIN_REVIEWS_FUTURE = 3      # need some reviews in the horizon to define a label
MIN_REVIEWS_PAST = 6
# Time-based split, derived from the data: the last TEST_MONTHS labelable months
# are the test period, the VALID_MONTHS before them validation, the rest training.
TEST_MONTHS = 18
VALID_MONTHS = 6
# A label at month t looks HORIZON_MONTHS ahead, so the last labels before each split boundary would see
# ratings from the next period. They are dropped (a purge gap) so validation and test outcomes never leak back.
SPLIT_GAP_MONTHS = HORIZON_MONTHS
# Risk tiers are defined as *lift* over the typical restaurant's risk, not as fixed probabilities:
# real declines are rare (~6% of restaurant-months on Yelp), so a calibrated model almost never
# outputs 30%, yet 2x the typical restaurant's risk is a meaningful warning. "Typical" is the mean
# predicted risk of the active restaurants in the same month, so the risk and its reference come
# from the same calibrated model (the training base rate came from a different period).
RISK_MEDIUM_LIFT = 1.5
RISK_HIGH_LIFT = 2.0
PLANS_TOP_N = 20               # the agent always covers at least the N riskiest active restaurants

# --------------------------------------------------------------------------
# M2 - transformer
# --------------------------------------------------------------------------
TRANSFORMER_BASE = os.getenv("RADVIEW_TRANSFORMER_BASE", "distilbert-base-uncased")
# 128 tokens cut off ~40% of Yelp reviews (and ~28% of aspect mentions); 256 covers most reviews.
MAX_SEQ_LEN = int(os.getenv("RADVIEW_MAX_SEQ_LEN", "256"))

# --------------------------------------------------------------------------
# M5 - agent
# --------------------------------------------------------------------------
LLM_MODEL = os.getenv("RADVIEW_LLM_MODEL", "claude-opus-5-5")
# Which LLM drafts the plans when one is used:
#   "anthropic" - Claude through the Anthropic API (needs ANTHROPIC_API_KEY)
#   "open"      - a free open-weight model behind an OpenAI-compatible endpoint, by default
#                 Qwen2.5 7B served by Ollama on the same machine (no key, no per-token cost)
LLM_PROVIDER = os.getenv("RADVIEW_LLM_PROVIDER", "anthropic")
OPEN_LLM_URL = os.getenv("RADVIEW_OPEN_LLM_URL", "http://localhost:11434/v1")
OPEN_LLM_MODEL = os.getenv("RADVIEW_OPEN_LLM_MODEL", "qwen2.5:7b")
OPEN_LLM_TIMEOUT = int(os.getenv("RADVIEW_OPEN_LLM_TIMEOUT", "600"))  # seconds per call; CPU inference is slow
EVIDENCE_PER_ASPECT = 3
EVIDENCE_LOOKBACK_MONTHS = 4
# Plan refreshes requested through the API / dashboard use the rule-based planner unless this is "1":
# otherwise anyone who can reach the dashboard can trigger paid LLM calls.
ALLOW_LLM_REFRESH = os.getenv("RADVIEW_ALLOW_LLM_REFRESH", "0") == "1"

# --------------------------------------------------------------------------
# Compute device for the PyTorch models (M2 transformer, M4 GRU).
# Default is CPU; set RADVIEW_DEVICE=cuda to opt in to a GPU.
# --------------------------------------------------------------------------
DEVICE = os.getenv("RADVIEW_DEVICE", "cpu")

RANDOM_SEED = 42
