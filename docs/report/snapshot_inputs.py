"""Copy the pipeline outputs the report needs into docs/report/inputs.

The data/ folder is git-ignored (it holds Yelp review text), so the report's figures and tables are built
from these small snapshots instead: evaluation JSON files and the M4 test-set predictions (business IDs,
months, labels and model scores only - no review text).

    python docs/report/snapshot_inputs.py      # after re-running the pipeline
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DST = Path(__file__).resolve().parent / "inputs"
YELP = ROOT / "data" / "work" / "yelp" / "outputs"
SYN = ROOT / "data" / "work" / "synthetic" / "outputs"

FILES = {
    YELP / "m2_evaluation.json": "m2_evaluation.json",                    # Table IV, Fig. 4
    YELP / "m3_evaluation.json": "m3_evaluation.json",                    # Table V
    YELP / "m4_evaluation.json": "m4_evaluation.json",                    # Table VI, Sec. IV-D
    YELP / "m4_test_predictions.parquet": "m4_test_predictions.parquet",  # Fig. 5 (PR curves)
    YELP / "m4_evaluation_test_end_2019-12.json": "m4_evaluation_test_end_2019-12.json",  # Table VII (pre-COVID)
    YELP / "m5_planner_comparison.json": "m5_planner_comparison.json",    # Table VIII
    SYN / "m4_evaluation.json": "m4_evaluation_synthetic.json",           # injected-episode results
    YELP / "m5_evaluation.json": "m5_evaluation.json",                    # M5 batch groundedness
    YELP / "pipeline_run.json": "pipeline_run.json",                      # module run times
    ROOT / "data" / "work" / "yelp" / "processed" / "ingest_report.json": "ingest_report.json",  # Table II
}

if __name__ == "__main__":
    DST.mkdir(exist_ok=True)
    for src, name in FILES.items():
        shutil.copy(src, DST / name)
        print(f"{src.relative_to(ROOT)} -> docs/report/inputs/{name}")
