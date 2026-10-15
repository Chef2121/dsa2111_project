"""Render the pipeline's evaluation JSON files into docs/RESULTS.md.

Every number in RESULTS.md comes straight from data/work/<source>/outputs, so
the report tables can be regenerated after any re-run.

Usage:  python scripts/make_results.py            # all sources that have outputs
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
OUT = ROOT / "docs" / "RESULTS.md"
MODEL_LABELS = {"majority": "Majority (base rate)", "current_negativity": "Current negativity (no model)",
                "rating_logreg": "Rating trend · logistic regression", "rating_lgbm": "Rating trend · LightGBM",
                "sentiment_lgbm": "Rating + overall sentiment · LightGBM", "radview_lgbm": "Radview features · LightGBM",
                "radview_gru": "**Radview · PyTorch GRU (ensemble, calibrated)**"}


def _load(src: str, name: str):
    f = WORK / src / "outputs" / name
    return json.loads(f.read_text()) if f.exists() else None


def _num(v):
    """Plain number (already in the unit it is shown in), or a dash."""
    return "–" if v is None else f"{v:g}"


def _fmt(v, pct=False):
    if v is None:
        return "–"
    return f"{v:.1%}" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def section(src: str) -> list[str]:
    lines = [f"## Data source: `{src}`", ""]
    if src == "synthetic":
        lines += ["_Simulator data: these numbers show the pipeline works end to end and test it against known, "
                  "injected deterioration episodes. They are not evidence about real restaurants._", ""]
    ingest = WORK / src / "processed" / "ingest_report.json"
    if ingest.exists():
        r = json.loads(ingest.read_text())
        lines += [f"{r['restaurants']} restaurants · {r['reviews_final']:,} reviews · "
                  f"{r['date_min'][:10]} → {r['date_max'][:10]} · mean rating {r['mean_stars']}", ""]

    m2 = _load(src, "m2_evaluation.json")
    if m2:
        lines += ["### M2 – aspect sentiment classification", ""]
        for bench, res in m2["results"].items():
            info = m2["benchmarks"][bench]
            lines += [f"Benchmark **{bench}** – {info['n']} labelled texts, aspects: {', '.join(info['aspects'])}", "",
                      "| Model | Macro-F1 | Complaint-F1 | ms / text |", "|---|---|---|---|"]
            for r in res:
                if "skipped" in r:
                    lines.append(f"| {r['model']} | – | – | skipped: {r['skipped']} |")
                else:
                    lines.append(f"| {r['model']} | {r['macro_f1']:.3f} | {r['complaint_f1']:.3f} | "
                                 f"{r['inference_ms_per_review']} |")
            lines.append("")
        lines += [f"Selected for production: **{m2['selected_model']}** (on {m2['selected_on']}).", ""]

    m3 = _load(src, "m3_evaluation.json")
    if m3:
        lines += ["### M3 – emerging complaint detection", "",
                  f"Base rate of a decline in the next 3 months: {m3['base_rate']:.1%}", "",
                  "| Detector | Flag rate | Precision | Lift | Decline onsets warned | Median lead (months) |",
                  "|---|---|---|---|---|---|"]
        for k in ("complaint_spike", "rating_alarm"):
            r = m3[k]
            lines.append(f"| {k.replace('_', ' ')} | {r['flag_rate']:.1%} | {r['precision']:.1%} | {r['lift']} | "
                         f"{r['onsets_detected_pct']}% of {r['onsets']} | {r['median_lead_months']} |")
        lines.append("")

    m4 = _load(src, "m4_evaluation.json")
    if m4:
        sp = m4["split"]
        lines += ["### M4 – rating-decline forecasting (out-of-time test)", "",
                  f"Train {sp['train']} · validation {sp['valid']} · test {sp['test']} — "
                  f"{sp['n_test']:,} test business-months, {sp['pos_rate_test']:.1%} positive.", "",
                  "| Model | AUROC | PR-AUC | Brier ↓ | Precision@10% | Recall@10% | Onsets warned (15% alert budget) | Median lead |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, r in m4["results"].items():
            ew = r.get("early_warning") or {}
            lines.append(f"| {MODEL_LABELS.get(k, k)} | {r['auroc']:.3f} | {r['pr_auc']:.3f} | {_fmt(r.get('brier'))} | "
                         f"{r['precision_at_10pct']:.3f} | {r['recall_at_10pct']:.3f} | {_num(ew.get('detected_pct'))}% | "
                         f"{_num(ew.get('median_lead_months'))} mo |")
        if m4.get("injected_episodes"):
            lines += ["", "Injected deterioration episodes (synthetic ground truth):", "",
                      "| Model | Episodes | Detected | Before visible drop | Median lead (months) |", "|---|---|---|---|---|"]
            for k, r in m4["injected_episodes"].items():
                lines.append(f"| {MODEL_LABELS.get(k, k)} | {r['episodes']} | {_num(r['detected_pct'])}% | "
                             f"{_num(r['before_visible_pct'])}% | {_num(r['median_lead_months'])} |")
        agr = m4.get("attribution_agreement")
        if agr:
            lines += ["", f"SHAP vs counterfactual — same top aspect: {_fmt(agr['all'].get('top_aspect_agreement'), True)} "
                      f"of {agr['all'].get('n')} restaurants ({_fmt(agr['at_risk'].get('top_aspect_agreement'), True)} "
                      f"of {agr['at_risk'].get('n')} at-risk); mean Spearman {_fmt(agr['all'].get('mean_spearman'))}.",
                      "Star-impact coefficients (stars per unit complaint rate): " +
                      ", ".join(f"{a} {c:+.2f}" for a, c in agr["star_impact_coefficients"].items())]
        lines += ["", "Top LightGBM features by mean |SHAP|: " +
                  ", ".join(f"`{k}`" for k in list(m4.get("lgbm_top_features_shap", {}))[:8]), ""]

    m5 = _load(src, "m5_evaluation.json")
    if m5:
        lines += ["### M5 – grounded action plans", "",
                  f"{m5['plans']} plans · {m5['claims_verified']}/{m5['claims_total']} drafted claims verified "
                  f"(groundedness **{_fmt(m5['groundedness'], True)}**) · planners: {m5.get('planners', m5.get('modes'))}", ""]
    comp = _load(src, "m5_planner_comparison.json")
    if comp:
        lines += [f"Planner comparison on the {comp['restaurants']} riskiest restaurants:", "",
                  "| Planner | Groundedness | Claims | Rejected: citation / quotation / relevance / statistic / substance | Items per plan | Median latency (s) | Tool calls |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in comp["planners"].items():
            f = v["failures_by_axis"]
            lines.append(f"| {k} | {_fmt(v['groundedness'], True)} | {v['claims_total']} | "
                         f"{f.get('citation', 0)} / {f.get('quotation', 0)} / {f.get('relevance', 0)} / "
                         f"{f.get('statistic', 0)} / {f.get('substance', 0)} | {v['mean_items_per_plan']} | "
                         f"{v['median_latency_s']} | {v['mean_tool_calls']} |")
        if comp.get("note"):
            lines += ["", f"_{comp['note']}_"]
        lines.append("")
    return lines


def main():
    lines = ["# Radview – experiment results", "",
             "_Generated by `scripts/make_results.py` from the pipeline's evaluation outputs._", ""]
    if not (WORK / "yelp" / "outputs" / "m4_evaluation.json").exists():
        lines += ["> **Real Yelp results pending** - this file is regenerated when the Yelp pipeline run finishes.", ""]
    for src in ("yelp", "synthetic"):
        if (WORK / src / "outputs" / "m4_evaluation.json").exists():
            lines += section(src)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
