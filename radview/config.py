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
SOURCE = os.getenv("RADVIEW_SOURCE") or (
    "yelp" if (_YELP_DIR / "yelp_academic_dataset_review.json").exists() else "synthetic")
RAW_DIR = Path(os.getenv("RADVIEW_RAW_DIR", _YELP_DIR if SOURCE == "yelp" else DATA_DIR / "synthetic" / "raw"))
WORK_DIR = DATA_DIR / "work" / SOURCE
PROCESSED_DIR = WORK_DIR / "processed"  # cleaned parquet tables, panel, trends
MODELS_DIR = WORK_DIR / "models"        # trained M2 / M4 artefacts
OUTPUTS_DIR = WORK_DIR / "outputs"      # scores, plans, evaluation reports
LABELS_DIR = DATA_DIR / "labels" / SOURCE  # hand-labelled aspect test set
SEMEVAL_FILE = DATA_DIR / "semeval" / "Restaurants_Test_Gold.xml"  # human-labelled ABSA benchmark

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, LABELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Slice of the real dataset used by default (CPU-friendly; override via env).
YELP_CITY = os.getenv("RADVIEW_CITY", "Philadelphia")
YELP_MAX_RESTAURANTS = int(os.getenv("RADVIEW_MAX_RESTAURANTS", "400"))
YELP_START_DATE = os.getenv("RADVIEW_START_DATE", "2016-01-01")  # older reviews are not needed for 2016+ monitoring

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
SPIKE_MIN_MENTIONS = 2   # need at least this many negative mentions in the month

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
RISK_HIGH = 0.30            # calibrated probability above which a restaurant is "high risk"
RISK_MEDIUM = 0.15

# --------------------------------------------------------------------------
# M2 - transformer
# --------------------------------------------------------------------------
TRANSFORMER_BASE = os.getenv("RADVIEW_TRANSFORMER_BASE", "distilbert-base-uncased")
MAX_SEQ_LEN = 128

# --------------------------------------------------------------------------
# M5 - agent
# --------------------------------------------------------------------------
LLM_MODEL = os.getenv("RADVIEW_LLM_MODEL", "claude-opus-5-5")
EVIDENCE_PER_ASPECT = 3
EVIDENCE_LOOKBACK_MONTHS = 4

RANDOM_SEED = 42
