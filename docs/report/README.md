# Group 10 report

| File | What it is |
|---|---|
| `Group10-report-DSA2111.docx` | The report (IEEE format). |
| `inputs/` | Snapshots of the pipeline outputs the report's results come from (metrics and M4 test predictions, no review text). `scripts/export_test_outputs.py` builds `docs/TEST_OUTPUTS.md` from them. |
| `figs/` | Figure images. |

Where each result comes from:

| Report item | Source in `inputs/` | Produced by |
|---|---|---|
| Table VIII, Fig. 4 (M2, SemEval columns) | `m2_evaluation.json` | `python -m radview.m2_aspects.train` |
| Table IX, Fig. 5 (M3) | `m3_evaluation.json` | `python -m radview.pipeline` (M3 step) |
| Table X, Fig. 6, Fig. 7 (M4) | `m4_evaluation.json`, `m4_test_predictions.parquet` | `python -m radview.m4_forecast.train` |
| Table XI (pre-COVID check) | `m4_evaluation_test_end_2019-12.json` | `python -m radview.m4_forecast.train --test-end 2019-12` |
| Table XII (selection robustness) | `robust_select2016_*.json` | `RADVIEW_DATA_DIR=data_select2016 python -m radview.pipeline --from-processed` (see docs/GPU_RERUN.md) |
| Table XIII (M5) | `m5_planner_comparison.json` | `python -m radview.m5_agent.cli --compare 20` |
| Injected-episode results (Sec. IV-D) | `m4_evaluation_synthetic.json` | `RADVIEW_SOURCE=synthetic python -m radview.m4_forecast.train` |

The Yelp columns of Table VIII come from the group's 300 hand-labelled reviews, which contain Yelp review text and
are therefore kept outside the repository.

All runs use fixed random seeds. Re-running them on the trained-models bundle reproduces the numbers, apart from
small differences between training runs on a GPU (Table VIII reports the spread for DistilBERT).
