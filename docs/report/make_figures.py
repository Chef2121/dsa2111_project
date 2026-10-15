"""Figures for the Group 10 report.

Fig. 1 architecture and Fig. 2 agent workflow are drawn here; Fig. 3 (dashboard) is composed from the two
screenshots in figs/src; Fig. 4 (M2 per-aspect F1) and Fig. 5 (M4 precision-recall) are computed from the
result snapshots in docs/report/inputs. Run:  python docs/report/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
# Inputs: the committed snapshots in docs/report/inputs (so figures rebuild from the repo alone);
# run snapshot_inputs.py after a new pipeline run to refresh them from data/work/yelp/outputs.
YELP = Path(__file__).resolve().parent / "inputs"
SRC = OUT / "src"   # raw dashboard screenshots for Fig. 5
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300"]


def box(ax, x, y, w, h, text, fc, fs=7, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec="#52514e", lw=0.7))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs, linespacing=1.25,
            fontweight="normal")
    if bold_first:
        pass


def arrow(ax, a, b, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#52514e",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.015, text, fontsize=6, ha="center", color="#52514e", style="italic")


# ---------------------------------------------------------------- Fig 1: architecture
fig, ax = plt.subplots(figsize=(7.0, 2.7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.165, 0.03), 0.825, 0.94, boxstyle="round,pad=0.005,rounding_size=0.02", fc="none",
                            ec="#2a78d6", lw=1, ls="--"))
ax.text(0.18, 0.93, "Kubernetes cluster (Docker containers, shared persistent volume)", color="#2a78d6", fontsize=7)
box(ax, 0.005, 0.56, 0.14, 0.3, "Data sources\nYelp Open Dataset\n(reviews, businesses,\ncheck-ins, tips)", "#e8f0fb")
box(ax, 0.005, 0.16, 0.14, 0.3, "Benchmark\nSemEval-2014\nTask 4 (restaurant\naspect labels)", "#e8f0fb")
box(ax, 0.185, 0.47, 0.15, 0.38, "M1 Ingestion\nstream 5 GB JSON,\nfilter restaurants,\nclean, de-duplicate\n→ parquet", "#e8f5ec")
box(ax, 0.375, 0.66, 0.17, 0.22, "M2 Aspect classifier\nfine-tuned DistilBERT\n(weak supervision)", "#fdf1e3")
box(ax, 0.375, 0.39, 0.17, 0.2, "M3 Trend detector\nEWMA + robust z-score", "#fdf1e3")
box(ax, 0.375, 0.08, 0.17, 0.24, "M4 Decline forecaster\nGRU ensemble, market\nfeatures, calibration,\nSHAP + counterfactual", "#fdf1e3")
box(ax, 0.59, 0.25, 0.2, 0.6, "M5 Agent (LangGraph)\n\nTools: query_metrics,\nrun_forecast,\nget_evidence_reviews,\ncompare_peers\n\nClaude / rule-based\nplanner → 5-axis\nclaim verifier", "#f1e8f8")
box(ax, 0.83, 0.25, 0.15, 0.6, "M6 Serving\n\nFastAPI\nREST API\n+\nStreamlit\nowner\ndashboard", "#e3f3f3")
box(ax, 0.185, 0.08, 0.15, 0.25, "Weekly CronJob\nre-ingest, re-score,\nregenerate plans", "#e8f5ec")
arrow(ax, (0.145, 0.71), (0.185, 0.68))
# SemEval feeds M2's evaluation: route through the gap between the M1 and M2 columns
ax.plot([0.145, 0.355, 0.355], [0.40, 0.40, 0.70], color="#52514e", lw=0.7)
arrow(ax, (0.355, 0.70), (0.375, 0.70))
ax.text(0.205, 0.415, "evaluation", fontsize=6, ha="center", color="#52514e", style="italic")
arrow(ax, (0.335, 0.72), (0.375, 0.77)); arrow(ax, (0.46, 0.66), (0.46, 0.59)); arrow(ax, (0.46, 0.39), (0.46, 0.32))
arrow(ax, (0.545, 0.77), (0.59, 0.72)); arrow(ax, (0.545, 0.2), (0.59, 0.33))
arrow(ax, (0.545, 0.49), (0.59, 0.5)); arrow(ax, (0.79, 0.55), (0.83, 0.55))
arrow(ax, (0.335, 0.2), (0.375, 0.2)); arrow(ax, (0.26, 0.33), (0.26, 0.47))
fig.savefig(OUT / "fig1_architecture.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: agent workflow
fig, ax = plt.subplots(figsize=(3.45, 2.05))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
steps = [("gather\nmetrics, forecast,\npeers", 0.02), ("prioritise\nimpact × urgency\n÷ effort", 0.27),
         ("retrieve\nevidence\nreviews", 0.52), ("draft\nClaude tool loop\nor rule-based", 0.77)]
for t, x in steps:
    box(ax, x, 0.58, 0.21, 0.32, t, "#f1e8f8", fs=6.2)
for i in range(3):
    arrow(ax, (steps[i][1] + 0.21, 0.74), (steps[i + 1][1], 0.74))
box(ax, 0.40, 0.06, 0.34, 0.32, "verify (5 axes)\ncitation · quotation ·\nrelevance · statistic ·\nsubstance", "#fdecec", fs=6.2)
box(ax, 0.02, 0.06, 0.3, 0.32, "finalise\nkeep verified claims;\nlog rejected ones;\ngroundedness score", "#e8f5ec", fs=6.2)
arrow(ax, (0.83, 0.58), (0.74, 0.30)); arrow(ax, (0.40, 0.22), (0.32, 0.22))
arrow(ax, (0.74, 0.16), (0.93, 0.58), rad=0.35)
ax.text(0.97, 0.30, "redraft\n(Claude,\n≤2 tries)", fontsize=5.8, ha="center", color="#52514e", style="italic")
fig.savefig(OUT / "fig2_agent.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: M2 per-aspect F1
m2 = json.loads((YELP / "m2_evaluation.json").read_text())["results"]["semeval2014"]
names = {"lexicon_rule": "Lexicon rule", "tfidf_logreg": "TF-IDF + LR", "distilbert_finetuned": "DistilBERT (ours)"}
aspects = ["food", "service", "price", "ambience"]
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
for ax, key, title in zip(axes, ["macro_f1_per_aspect", "complaint_f1_per_aspect"], ["Macro-F1 (3 classes)", "Complaint F1"]):
    w = 0.26
    for i, r in enumerate([r for r in m2 if r["model"] in names]):
        ax.bar([j + (i - 1) * w for j in range(4)], [r[key][a] for a in aspects], w, label=names[r["model"]],
               color=SER[i], edgecolor="white", lw=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["food", "service", "price", "amb."]); ax.set_title(title); ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("F1 on SemEval-2014")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.42), ncol=3, frameon=False)
fig.savefig(OUT / "fig3_m2.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: M4 PR curves
tp = pd.read_parquet(YELP / "m4_test_predictions.parquet")
labels = {"rating_logreg": "Rating trend · LogReg", "rating_lgbm": "Rating trend · LightGBM",
          "sentiment_lgbm": "+ sentiment · LightGBM", "radview_lgbm": "All features · LightGBM",
          "radview_gru": "All features · GRU (ours)", "current_negativity": "Current negativity"}
fig, ax = plt.subplots(figsize=(3.45, 2.3))
for i, (k, lab) in enumerate(labels.items()):
    p, r, _ = precision_recall_curve(tp["decline"], tp[k])
    ax.plot(r, p, lw=1.6 if k == "radview_gru" else 1.0, color=SER[i], label=lab,
            ls="--" if k == "current_negativity" else "-")
ax.axhline(tp["decline"].mean(), color="#52514e", lw=0.7, ls=":", label=f"Base rate ({tp['decline'].mean():.1%})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
ax.grid(color="#e6e5e0", lw=0.5); ax.legend(frameon=False, fontsize=6)
fig.savefig(OUT / "fig4_m4_pr.png"); plt.close(fig)
# ---------------------------------------------------------------- Fig 5 (report Fig. 3): dashboard screenshot
from PIL import Image

top = Image.open(SRC / "dashboard_top.jpg").crop((205, 110, 760, 478))      # verdict, KPIs, charts
plan = Image.open(SRC / "dashboard_plan.jpg").crop((200, 285, 755, 563))    # top of the action plan
dash = Image.new("RGB", (max(top.width, plan.width), top.height + plan.height + 8), "white")
dash.paste(top, (0, 0))
dash.paste(plan, (0, top.height + 8))
dash.save(OUT / "fig5_dashboard.png", dpi=(220, 220))
print("figures ->", OUT)
