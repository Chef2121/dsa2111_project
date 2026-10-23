"""Download the datasets used by Radview.

* Yelp Open Dataset (Kaggle mirror "adamamer2001/yelp-complete-open-dataset-2024"):
  business, review, checkin and tip JSON files (~5.9 GB unzipped) -> data/yelp/
  By using it you accept Yelp's Dataset User Agreement (academic, non-commercial use).
* SemEval-2014 Task 4 restaurant gold test set + training set (HuggingFace mirror) -> data/semeval/

Usage:  python scripts/download_data.py [--skip-yelp] [--skip-semeval]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YELP_DIR = ROOT / "data" / "yelp"
SEMEVAL_DIR = ROOT / "data" / "semeval"
KAGGLE_OWNER, KAGGLE_DATASET = "adamamer2001", "yelp-complete-open-dataset-2024"
YELP_FILES = ["business", "review", "checkin", "tip"]


def download_yelp() -> None:
    from kagglesdk import KaggleClient
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    YELP_DIR.mkdir(parents=True, exist_ok=True)
    client = KaggleClient()
    for name in YELP_FILES:
        target = YELP_DIR / f"yelp_academic_dataset_{name}.json"
        if target.exists():
            print(f"[yelp] {target.name} already present")
            continue
        req = ApiDownloadDatasetRequest()
        req.owner_slug, req.dataset_slug = KAGGLE_OWNER, KAGGLE_DATASET
        req.file_name = f"yelp_dataset/{target.name}"
        print(f"[yelp] downloading {target.name} ...", flush=True)
        resp = client.datasets.dataset_api_client.download_dataset(req)
        archive = YELP_DIR / f"{name}.zip"
        with open(archive, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
        with zipfile.ZipFile(archive) as zf:  # Kaggle serves single files zipped
            zf.extractall(YELP_DIR)
        archive.unlink()


def download_semeval() -> None:
    from huggingface_hub import hf_hub_download

    SEMEVAL_DIR.mkdir(parents=True, exist_ok=True)
    # the training file supplies the dev set M2 selects its production model on (the test set stays unseen)
    train = hf_hub_download("alexcadillon/SemEval2014Task4",
                            "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml",
                            repo_type="dataset")
    shutil.copy(train, SEMEVAL_DIR / "Restaurants_Train_v2.xml")
    print("[semeval] saved Restaurants_Train_v2.xml")
    path = hf_hub_download("alexcadillon/SemEval2014Task4", "ABSA_Gold_TestData/Restaurants_Test_Gold.xml",
                           repo_type="dataset")
    shutil.copy(path, SEMEVAL_DIR / "Restaurants_Test_Gold.xml")
    print("[semeval] saved Restaurants_Test_Gold.xml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-yelp", action="store_true")
    ap.add_argument("--skip-semeval", action="store_true")
    args = ap.parse_args()
    if not args.skip_semeval:
        download_semeval()
    if not args.skip_yelp:
        download_yelp()
    sys.exit(0)
