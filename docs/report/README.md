# Group 10 report — how it is built

| File | What it is |
|---|---|
| `Group10-report-DSA2111.docx` | The report (IEEE format). Yellow-highlighted parts are for the group to complete. |
| `build_report.py` | Builds the .docx. Every results table is read from `inputs/`. |
| `make_figures.py` | Draws Figs. 1–2, composes Fig. 3 from `figs/src/`, computes Figs. 4–7 from `inputs/`. |
| `inputs/` | Snapshots of the pipeline outputs the report uses (metrics and M4 test predictions — no review text). |
| `snapshot_inputs.py` | Refreshes `inputs/` from `data/work/*/outputs` after a new pipeline run. |
| `figs/` | Generated figures; `figs/src/` holds the two raw dashboard screenshots behind Fig. 3. |

Rebuild (no data folder needed):

```bash
python docs/report/make_figures.py
python docs/report/build_report.py            # or: build_report.py other-name.docx
```

Where each table/figure comes from:

| Report item | Source in `inputs/` | Produced by |
|---|---|---|
| Table VIII, Fig. 4 (M2) | `m2_evaluation.json` | `python -m radview.m2_aspects.train` |
| Table IX, Fig. 5 (M3) | `m3_evaluation.json` | `python -m radview.pipeline` (M3 step) |
| Table X, Fig. 6, Fig. 7 (M4) | `m4_evaluation.json`, `m4_test_predictions.parquet` | `python -m radview.m4_forecast.train` |
| Table XI (pre-COVID check) | `m4_evaluation_test_end_2019-12.json` | `python -m radview.m4_forecast.train --test-end 2019-12` |
| Table XII (selection robustness) | `robust_select2016_*.json` | `RADVIEW_DATA_DIR=data_select2016 python -m radview.pipeline --from-processed` (see docs/GPU_RERUN.md) |
| Table XIII (M5) | `m5_planner_comparison.json` | `python -m radview.m5_agent.cli --compare 20` |
| Injected-episode results (Sec. IV-D) | `m4_evaluation_synthetic.json` | `RADVIEW_SOURCE=synthetic python -m radview.m4_forecast.train` |

All runs use fixed random seeds; re-running them on the trained-models bundle reproduces the numbers exactly.
