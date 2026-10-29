"""Robustness check: compare the headline results of two pipeline runs side by side.

Meant for the restaurant-selection check: the main analysis (the 400 restaurants with the most lifetime
reviews) against the same pipeline on the 400 with the most reviews in 2016 (RADVIEW_SELECT_BY=early),
which uses no information from after the start of the study period.

Usage:  python scripts/compare_selection.py --main data --robust data_select2016 [--out docs/ROBUSTNESS_SELECTION.md]
        (each argument is a RADVIEW_DATA_DIR; results are read from <dir>/work/yelp/)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

M4_MODELS = ["majority", "current_negativity", "rating_logreg", "rating_lgbm", "sentiment_lgbm", "radview_lgbm",
             "radview_gru"]


def _json(path: Path):
    """Parsed JSON file, or {} if it does not exist."""
    return json.loads(path.read_text()) if path.exists() else {}


def summarise(data_dir: Path) -> dict[str, str]:
    """Headline results of one restaurant selection (data directory) as label -> formatted value."""
    work = data_dir / "work" / "yelp"
    out, proc = work / "outputs", work / "processed"
    biz = pd.read_parquet(proc / "businesses.parquet")
    ingest = _json(proc / "ingest_report.json")
    rows = {"Restaurants (still open)": f"{len(biz)} ({biz['is_open'].mean():.0%})",
            "Reviews": f"{ingest.get('reviews_final', 0):,}"}

    m2 = _json(out / "m2_evaluation.json")
    sel = m2.get("selected_model")
    best = next((r for r in m2.get("results", {}).get("semeval2014", []) if r.get("model") == sel), {})
    rows["M2 selected model (on " + str(m2.get("selected_on")) + ")"] = str(sel)
    rows["M2 macro-F1 / complaint-F1 (SemEval test)"] = f"{best.get('macro_f1')} / {best.get('complaint_f1')}"

    m3 = _json(out / "m3_evaluation.json")
    cs = m3.get("complaint_spike", {})
    ch = cs.get("chance", {})
    rows["M3 decline base rate"] = f"{m3.get('base_rate', float('nan')):.2%}"
    rows["M3 spike lift (random timing)"] = f"{cs.get('lift')} ({ch.get('lift')})"
    rows["M3 onsets detected % (random timing)"] = f"{cs.get('onsets_detected_pct')} ({ch.get('onsets_detected_pct')})"
    rows["M3 lasting-drop lift (random timing)"] = (f"{(cs.get('lasting_drop') or {}).get('lift')} "
                                                   f"({ch.get('lasting_drop_lift')})")

    m4 = _json(out / "m4_evaluation.json")
    res = m4.get("results", {})
    for name in M4_MODELS:
        r = res.get(name)
        if r:
            rows[f"M4 {name}: AUROC / PR-AUC"] = f"{r['auroc']} / {r['pr_auc']}"
    learned = {k: v for k, v in res.items() if k not in ("majority", "current_negativity")}
    if learned:
        rows["M4 best model by PR-AUC"] = max(learned, key=lambda k: learned[k]["pr_auc"])
    ew = (res.get("radview_gru") or {}).get("early_warning") or {}
    rows["M4 GRU onsets warned % (random alerts)"] = f"{ew.get('detected_pct')} ({ew.get('chance_detected_pct')})"

    comp = _json(out / "m5_planner_comparison.json").get("planners", {})
    for name, v in comp.items():
        rows[f"M5 groundedness: {name}"] = f"{v.get('groundedness')} ({v.get('claims_verified')}/{v.get('claims_total')})"
    return rows


def main() -> None:
    """Print or write the side-by-side comparison of the main and robustness selections."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, default=Path("data"))
    ap.add_argument("--robust", type=Path, default=Path("data_select2016"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    a, b = summarise(args.main), summarise(args.robust)
    lines = ["# Robustness check: restaurant selection", "",
             "Main analysis = the 400 Philadelphia restaurants with the most lifetime Yelp reviews; robustness = the 400",
             "with the most reviews in 2016 (no information from after the start of the study period).", "",
             "| Result | Main (lifetime top 400) | Robustness (2016 top 400) |", "|---|---|---|"]
    lines += [f"| {k} | {a.get(k, '–')} | {b.get(k, '–')} |" for k in dict.fromkeys([*a, *b])]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
