"""Collect every test output of the project into one file: docs/TEST_OUTPUTS.md.

Contents
    1. Index - each experiment, where it appears in the report, the command that produced it, its source file
    2. Dataset (M1) and run times
    3. M2 aspect classification - every model, benchmark, aspect and metric
    4. M3 complaint-spike detection
    5. M4 decline forecasting - main (COVID-era) test, pre-COVID test (Table VII), synthetic injected episodes,
       GRU training curves, attribution agreement, star-impact model, top features
    6. M5 action plans - batch groundedness and planner comparison
    7. Automated tests - the result of every pytest test, run now
    8. Appendix - the raw result files (collapsible)

Sources are the committed snapshots in docs/report/inputs/, so this file can be regenerated from the repo.
The pytest section runs the suite; agent/API tests need the trained-models data (RADVIEW_DATA_DIR or ./data)
and are reported as skipped without it.

    python scripts/export_test_outputs.py
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "docs" / "report" / "inputs"
OUT = ROOT / "docs" / "TEST_OUTPUTS.md"

MODEL = {"majority": "Majority (base rate)", "current_negativity": "Current negativity (no model)",
         "rating_logreg": "Rating trend · logistic regression", "rating_lgbm": "Rating trend · LightGBM",
         "sentiment_lgbm": "+ overall sentiment · LightGBM", "radview_lgbm": "All features · LightGBM",
         "radview_gru": "**All features · PyTorch GRU (Radview)**",
         "lexicon_rule": "Lexicon rule (weak labeller)", "tfidf_logreg": "TF-IDF + logistic regression",
         "distilbert_finetuned": "**DistilBERT, fine-tuned (Radview)**", "zero_shot_llm": "Zero-shot LLM (Claude)"}


def load(name: str):
    return json.loads((INP / name).read_text(encoding="utf-8"))


def n(v, fmt="{:.3f}"):
    if v is None:
        return "–"
    return fmt.format(v) if isinstance(v, (int, float)) else str(v)


def pct(v):
    return "–" if v is None else f"{v:.1%}"


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------------- sections
def section_index() -> list[str]:
    rows = [
        ["Dataset after ingestion (M1)", "Table II", "`python -m radview.m1_ingest.ingest`", "`ingest_report.json`"],
        ["Aspect classification (M2)", "Table IV, Fig. 4", "`python -m radview.m2_aspects.train`", "`m2_evaluation.json`"],
        ["Complaint-spike detection (M3)", "Table V", "`python -m radview.pipeline` (M3 step)", "`m3_evaluation.json`"],
        ["Decline forecasting, main test 2020-04–2021-09 (M4)", "Table VI, Fig. 5, Table VII left",
         "`python -m radview.m4_forecast.train`", "`m4_evaluation.json`, `m4_test_predictions.parquet`"],
        ["Decline forecasting, pre-COVID test 2018-07–2019-12", "Table VII right",
         "`python -m radview.m4_forecast.train --test-end 2019-12`", "`m4_evaluation_test_end_2019-12.json`"],
        ["Injected deterioration episodes (synthetic)", "Sec. IV-D",
         "`RADVIEW_SOURCE=synthetic python -m radview.m4_forecast.train`", "`m4_evaluation_synthetic.json`"],
        ["Action plans, batch (M5)", "Sec. IV-G", "`python -m radview.m5_agent.agent`", "`m5_evaluation.json`"],
        ["Planner comparison (M5)", "Table VIII", "`python -m radview.m5_agent.cli --compare 20`",
         "`m5_planner_comparison.json`"],
        ["Automated tests", "Sec. IV-H", "`python -m pytest -v`", "run when this file is generated"],
    ]
    return ["## 1. Index", "", "All source files are in [`docs/report/inputs/`](report/inputs/). Every run uses fixed "
            "random seeds; re-running a command on the trained-models bundle reproduces its numbers exactly.", ""] + \
        table(["Experiment", "In the report", "Command", "Source file"], rows)


def section_dataset() -> list[str]:
    r = load("ingest_report.json")
    run = load("pipeline_run.json")
    rows = [["Restaurants", r["restaurants"]], ["Review records read", f"{r['rows_read']:,}"],
            ["Reviews of the selected restaurants", f"{r['rows_kept_restaurants']:,}"],
            ["Removed: before start date / invalid / empty", f"{r['dropped_invalid_or_empty']:,}"],
            ["Removed: duplicates", f"{r['dropped_duplicates']:,}"],
            ["Removed: partial last month", f"{r['dropped_partial_last_month']:,}"],
            ["**Reviews used**", f"**{r['reviews_final']:,}**"],
            ["Period", f"{r['date_min'][:10]} → {r['date_max'][:10]}"], ["Mean star rating", r["mean_stars"]],
            ["Business-months with check-ins / tips", f"{r['checkin_business_months']:,} / {r['tip_business_months']:,}"]]
    t = run["timings_s"]
    times = [[m.upper(), f"{s:,.0f} s ({s / 60:.1f} min)"] for m, s in t.items()]
    return ["## 2. Dataset (M1) and run times", "", "Yelp Open Dataset, 400 most-reviewed Philadelphia restaurants.", ""] + \
        table(["Quantity", "Value"], rows) + \
        [f"Full pipeline run ({run['source']}, mode `{run['mode']}`, finished {run['finished_at']}), CPU only:", ""] + \
        table(["Module", "Time"], times)


def section_m2() -> list[str]:
    m2 = load("m2_evaluation.json")
    out = ["## 3. M2 — aspect-sentiment classification", "",
           f"Selected for production: **{m2['selected_model']}** (by macro-F1 on {m2['selected_on']}). "
           "Labels: 0 = not mentioned, 1 = positive, 2 = negative. Complaint-F1 = F1 of the negative class.", ""]
    for bench, res in m2["results"].items():
        info = m2["benchmarks"][bench]
        aspects = info["aspects"]
        out += [f"### Benchmark `{bench}` — {info['n']} human-labelled texts", ""]
        rows = []
        for r in res:
            if "skipped" in r:
                rows.append([MODEL.get(r["model"], r["model"]), "–", "–", "–", f"skipped: {r['skipped']}"])
            else:
                extra = f"trained on {r['train_size']:,} reviews in {r['train_seconds']:,.0f} s" if r.get("train_seconds") else ""
                rows.append([MODEL.get(r["model"], r["model"]), n(r["macro_f1"]), n(r["complaint_f1"]),
                             n(r["inference_ms_per_review"], "{:.2f}"), extra])
        out += table(["Model", "Macro-F1", "Complaint-F1", "ms / text", "Note"], rows)
        for key, title in (("macro_f1_per_aspect", "Macro-F1 per aspect"), ("complaint_f1_per_aspect", "Complaint-F1 per aspect")):
            out += [f"*{title}:*", ""]
            out += table(["Model"] + aspects, [[MODEL.get(r["model"], r["model"])] + [n(r[key][a]) for a in aspects]
                                               for r in res if key in r])
    wd = m2["weak_label_distribution"]
    out += ["### Weak-label distribution (training sample)", ""]
    out += table(["Aspect", "Not mentioned", "Positive", "Negative"],
                 [[a, f"{v[0]:,}", f"{v[1]:,}", f"{v[2]:,}"] for a, v in wd.items()])
    return out


def section_m3() -> list[str]:
    m3 = load("m3_evaluation.json")
    rows = [[k.replace("_", " "), pct(v["flag_rate"]), pct(v["precision"]), n(v["lift"], "{:.2f}"),
             v["onsets"], f"{v['onsets_detected_pct']}%", n(v["median_lead_months"], "{:g}")]
            for k, v in m3.items() if isinstance(v, dict)]
    return ["## 4. M3 — emerging complaint detection", "",
            f"Does an alert this month anticipate a decline in the next 3 months? Base rate: {pct(m3['base_rate'])}. "
            "Lift = precision ÷ base rate (below 1 means worse than chance).", ""] + \
        table(["Detector", "Months flagged", "Precision", "Lift", "Decline onsets", "Onsets warned (≤6 mo)",
               "Median lead (months)"], rows)


def _m4_rows(results: dict) -> list[list]:
    rows = []
    for k, r in results.items():
        ew = r.get("early_warning") or {}
        rows.append([MODEL.get(k, k), n(r["auroc"]), n(r["pr_auc"]), n(r.get("brier"), "{:.4f}"),
                     n(r["precision_at_10pct"]), n(r["recall_at_10pct"]),
                     f"{ew['detected_pct']}%" if ew.get("detected_pct") is not None else "–",
                     n(ew.get("median_lead_months"), "{:g}"), ew.get("onsets", "–")])
    return rows


M4_HEAD = ["Model", "AUROC", "PR-AUC", "Brier ↓", "P@10%", "R@10%", "Onsets warned", "Median lead (mo)", "Onsets"]


def _split(sp: dict) -> str:
    return (f"Train {sp['train']} ({sp['n_train']:,} restaurant-months) · validation {sp['valid']} "
            f"({sp['n_valid']:,}) · test {sp['test']} ({sp['n_test']:,}); positive rate {pct(sp['pos_rate_train'])} "
            f"train / {pct(sp['pos_rate_test'])} test.")


def section_m4() -> list[str]:
    m4, pre, syn = load("m4_evaluation.json"), load("m4_evaluation_test_end_2019-12.json"), load("m4_evaluation_synthetic.json")
    out = ["## 5. M4 — rating-decline forecasting", "",
           "Label: next-3-month mean rating ≥ 0.3★ below the trailing 6 months *and* a significant drop (one-sided "
           "z ≥ 1.645). All models are Platt-calibrated on the validation months. P@10% / R@10% = precision / recall "
           "in the riskiest 10% of test restaurant-months; *Onsets warned* = decline onsets flagged within the previous "
           "6 months when each model may flag 15% of months.", "",
           "### 5.1 Main test window (COVID era) — Table VI", "", _split(m4["split"]), ""]
    out += table(M4_HEAD, _m4_rows(m4["results"]))
    out += ["### 5.2 Pre-COVID test window — Table VII (right column)", "",
            "Command: `python -m radview.m4_forecast.train --test-end 2019-12` (labels after Dec 2019 hidden; "
            "evaluation only, deployed models untouched).", "", _split(pre["split"]), ""]
    out += table(M4_HEAD, _m4_rows(pre["results"]))
    out += ["*Table VII as printed in the report (AUROC / PR-AUC):*", ""]
    out += table(["Model", "2020-04 – 2021-09", "2018-07 – 2019-12"],
                 [[MODEL.get(k, k), f"{m4['results'][k]['auroc']:.3f} / {m4['results'][k]['pr_auc']:.3f}",
                   f"{pre['results'][k]['auroc']:.3f} / {pre['results'][k]['pr_auc']:.3f}"]
                  for k in m4["results"] if k != "majority"])
    out += ["### 5.3 Synthetic data — injected deterioration episodes", "",
            "On the simulator the onset of each deterioration episode is known. *Detected* = an alert fired between "
            "onset and two months after the drop became visible; *before visible* = the alert fired before the "
            "3-month rating had dropped 0.3★.", "", _split(syn["split"]), ""]
    out += table(M4_HEAD, _m4_rows(syn["results"]))
    out += table(["Model", "Episodes", "Detected", "Before visible drop", "Median lead (months)"],
                 [[MODEL.get(k, k), v["episodes"], f"{v['detected_pct']}%", f"{v['before_visible_pct']}%",
                   n(v["median_lead_months"], "{:g}")] for k, v in syn["injected_episodes"].items()])
    gt = m4["gru_training"]
    out += ["### 5.4 GRU training (main run)", "",
            f"Ensemble of {len(gt)} networks; Platt calibration (a, b) = "
            f"({m4['calibration'][0]:.3f}, {m4['calibration'][1]:.3f}).", ""]
    out += table(["Seed", "Epochs run", "Best validation PR-AUC", "Validation PR-AUC per epoch"],
                 [[i, len(h["history"]), n(h["best_valid_pr_auc"], "{:.4f}"),
                   " ".join(f"{e['valid_pr_auc']:.3f}" for e in h["history"])] for i, h in enumerate(gt)])
    agr = m4["attribution_agreement"]
    out += ["### 5.5 Explanations — SHAP vs counterfactual, star-impact model", ""]
    out += table(["Restaurants", "n", "Same top aspect", "Mean Spearman"],
                 [[k, v["n"], pct(v["top_aspect_agreement"]), n(v["mean_spearman"], "{:.3f}")]
                  for k, v in agr.items() if isinstance(v, dict) and "n" in v])
    out += ["Star-impact ridge coefficients (stars per unit complaint rate):", ""]
    out += table(["Aspect", "Coefficient"], [[a, f"{c:+.3f}"] for a, c in agr["star_impact_coefficients"].items()])
    out += ["Top LightGBM features (all-features model, mean |SHAP| on the test set):", ""]
    out += table(["Feature", "Mean abs SHAP"], [[f"`{k}`", f"{v:.4f}"] for k, v in m4["lgbm_top_features_shap"].items()])
    tp = pd.read_parquet(INP / "m4_test_predictions.parquet")
    out += [f"Test-set predictions behind Fig. 5: `m4_test_predictions.parquet` — {len(tp):,} rows "
            f"({tp['business_id'].nunique()} restaurants × months), columns {', '.join(f'`{c}`' for c in tp.columns)}.", ""]
    return out


def section_m5() -> list[str]:
    b, c = load("m5_evaluation.json"), load("m5_planner_comparison.json")
    out = ["## 6. M5 — grounded action plans", "",
           "### 6.1 Weekly batch (all at-risk restaurants)", ""]
    out += table(["Plans", "Claims", "Verified", "Groundedness", "Planner(s)", "Median latency"],
                 [[b["plans"], b["claims_total"], b["claims_verified"], pct(b["groundedness"]),
                   ", ".join(f"{k} × {v}" for k, v in b["planners"].items()), f"{b['median_latency_s']} s"]])
    out += [f"### 6.2 Planner comparison — Table VIII ({c['restaurants']} riskiest restaurants)", "",
            "Rejections per verifier axis (one claim can fail several axes).", ""]
    out += table(["Planner", "Ran as", "Claims", "Verified", "Groundedness", "Citation", "Quotation", "Relevance",
                  "Statistic", "Substance", "Items / plan", "Median latency (s)", "Tool calls"],
                 [[k, ", ".join(v["actually_ran"]), v["claims_total"], v["claims_verified"], pct(v["groundedness"]),
                   *[v["failures_by_axis"][a] for a in ("citation", "quotation", "relevance", "statistic", "substance")],
                   v["mean_items_per_plan"], v["median_latency_s"], v["mean_tool_calls"]] for k, v in c["planners"].items()])
    if c.get("note"):
        out += [f"*{c['note']}*", ""]
    return out


def section_pytest() -> list[str]:
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider"], cwd=ROOT,
                          capture_output=True, text=True, env=env)
    lines = proc.stdout.splitlines()
    results = []
    for ln in lines:
        m = re.match(r"^(tests/\S+?)::(\S+)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)", ln)
        if m:
            results.append([f"`{m.group(1)}`", f"`{m.group(2)}`", m.group(3)])
    summary = next((ln.strip("= ").strip() for ln in reversed(lines) if re.search(r"\d+ (passed|failed|skipped)", ln)), "?")
    skips = [ln for ln in lines if ln.startswith("SKIPPED")]
    data_dir = os.getenv("RADVIEW_DATA_DIR", str(ROOT / "data"))
    out = ["## 7. Automated tests (pytest)", "",
           f"Run while generating this file: **{summary}** — Python {platform.python_version()}, {platform.system()}, "
           f"data directory `{Path(data_dir).name}/` {'present' if Path(data_dir, 'work').exists() else 'absent'}. "
           "Agent and API tests need the trained-models data and are skipped without it.", ""]
    out += table(["File", "Test", "Result"], results)
    if skips:
        out += ["Skip reasons:", "", "```", *skips, "```", ""]
    return out


def section_appendix() -> list[str]:
    out = ["## 8. Appendix — raw result files", "",
           "The exact JSON files the tables above were rendered from (training histories included).", ""]
    for f in sorted(INP.glob("*.json")):
        out += [f"<details><summary><code>{f.name}</code></summary>", "", "```json",
                json.dumps(json.loads(f.read_text(encoding="utf-8")), indent=2), "```", "", "</details>", ""]
    return out


def main():
    lines = ["# Radview — all test outputs", "",
             f"_Generated by `scripts/export_test_outputs.py` on {datetime.now():%Y-%m-%d %H:%M} at commit "
             f"`{git_commit()}`. Regenerate with `python scripts/export_test_outputs.py`._", "",
             "This file collects every evaluation result and test outcome of the project in one place. "
             "Headline numbers and their interpretation are in the report and in [FINDINGS.md](FINDINGS.md).", ""]
    for part in (section_index, section_dataset, section_m2, section_m3, section_m4, section_m5, section_pytest,
                 section_appendix):
        lines += part()
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
