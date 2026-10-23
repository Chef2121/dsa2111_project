"""Build Group10-report-DSA2111.docx (IEEE conference style, 10 pt Times New Roman, two columns).

Every results table is read from the snapshots of the pipeline's evaluation outputs in docs/report/inputs,
so numbers cannot drift from the code and the report rebuilds without the (git-ignored) data folder.
Run make_figures.py first; run snapshot_inputs.py after a new pipeline run to refresh the snapshots.

    python docs/report/build_report.py [output.docx]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGS = HERE / "figs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "Group10-report-DSA2111.docx"  # optional output path
Y = HERE / "inputs"   # committed snapshots of the pipeline outputs (see snapshot_inputs.py)
m2 = json.loads((Y / "m2_evaluation.json").read_text())
m3 = json.loads((Y / "m3_evaluation.json").read_text())
m4 = json.loads((Y / "m4_evaluation.json").read_text())
m5c = json.loads((Y / "m5_planner_comparison.json").read_text())
m5b = json.loads((Y / "m5_evaluation.json").read_text())
rob = {k: json.loads((Y / f"robust_select2016_{k}.json").read_text())
       for k in ("m2_evaluation", "m3_evaluation", "m4_evaluation", "m5_planner_comparison")}
m4s = json.loads((Y / "m4_evaluation_synthetic.json").read_text())
m4pre = json.loads((Y / "m4_evaluation_test_end_2019-12.json").read_text())
res2 = {r["model"]: r for r in m2["results"]["semeval2014"] if "macro_f1" in r}
LLM_SLOWDOWN = round(res2["zero_shot_llm"]["inference_ms_per_review"]
                     / res2["distilbert_finetuned"]["inference_ms_per_review"], -1) if "zero_shot_llm" in res2 else None

FONT = "Times New Roman"
COL_W = Inches(3.45)

doc = Document()

# ------------------------------------------------------------------ base styles
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
pf = st.paragraph_format
pf.space_before = Pt(0)
pf.space_after = Pt(0)
pf.line_spacing = 1.0

sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
sec.top_margin, sec.bottom_margin = Mm(19), Mm(25)
sec.left_margin = sec.right_margin = Mm(14.3)


def set_columns(section, num: int, space_in: float = 0.24):
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(space_in * 1440)))


_TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\[\[.+?\]\]|_\{.+?\})")


def add_runs(p, text: str, size: float | None = None, bold: bool = False, italic: bool = False):
    """Mini markup: **bold**, *italic*, _{subscript}, [[placeholder]] (yellow-highlighted, for the group to fill)."""
    for part in _TOKEN.split(text):
        if not part:
            continue
        r_bold, r_ital, hl, sub = bold, italic, False, False
        if part.startswith("_{"):
            part, sub, r_ital = part[2:-1], True, True
        elif part.startswith("**"):
            part, r_bold = part[2:-2], True
        elif part.startswith("[["):
            part, hl = part[2:-2], True
        elif part.startswith("*"):
            part, r_ital = part[1:-1], True
        r = p.add_run(part)
        r.bold, r.italic = r_bold, r_ital
        if size:
            r.font.size = Pt(size)
        if hl:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if sub:
            r.font.subscript = True
    return p


def para(text: str, indent: bool = True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, space_after=0, **kw):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.first_line_indent = Inches(0.17) if indent else None
    p.paragraph_format.space_after = Pt(space_after)
    return add_runs(p, text, size=size, **kw)


def bullets(items: list[str]):
    for it in items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.first_line_indent = Inches(-0.14)
        add_runs(p, "• " + it)


_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
          "XVII", "XVIII", "XIX", "XX"]
_h1 = [0]
_h2 = [0]


def h1(title: str, numbered: bool = True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    if numbered:
        _h1[0] += 1
        _h2[0] = 0
        title = f"{_ROMAN[_h1[0] - 1]}. {title}"
    r = p.add_run(title)
    r.font.small_caps = True
    r.font.size = Pt(10)


def h2(title: str):
    _h2[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{chr(64 + _h2[0])}. {title}")
    r.italic = True


_fig = [0]
_tab = [0]


def figure(path: Path, caption: str, width=COL_W):
    _fig[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=width)
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(6)
    add_runs(c, f"Fig. {_fig[0]}. ", size=8)
    add_runs(c, caption, size=8)
    return _fig[0]


def _borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge, val in (("top", top), ("bottom", bottom)):
        if val:
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(val))
            e.set(qn("w:color"), "000000")
            b.append(e)
    tcPr.append(b)


def table(caption: str, header: list[str], rows: list[list[str]], widths: list[float], size=7.5, note: str | None = None):
    """IEEE-style table: caption above (TABLE N + title in small caps), horizontal rules only."""
    _tab[0] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_before = Pt(6)
    c.paragraph_format.keep_with_next = True
    r = c.add_run(f"TABLE {_ROMAN[_tab[0] - 1]}")
    r.font.size = Pt(8)
    c2 = doc.add_paragraph()
    c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c2.paragraph_format.space_after = Pt(2)
    c2.paragraph_format.keep_with_next = True
    r = c2.add_run(caption)
    r.font.size = Pt(8)
    r.font.small_caps = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # Fixed layout so Word honours the column widths instead of re-flowing narrow two-column tables.
    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w * 1440)))
    mar = OxmlElement("w:tblCellMar")  # narrow cell padding: default 0.08" per side wraps short headers
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), "43")
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Inches(widths[j])
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, str(val), size=size, bold=(i == 0))
            p.paragraph_format.keep_with_next = i < len(rows)  # keep the table (and its caption) on one page
            if i == 0:
                _borders(cell, top=8, bottom=4)
            elif i == len(rows):
                _borders(cell, bottom=8)
    if note:
        n = doc.add_paragraph()
        n.paragraph_format.space_before = Pt(1)
        add_runs(n, note, size=7, italic=True)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)
    return _tab[0]


def wide(fn):
    """Temporarily switch to a single full-width column (for wide figures/tables)."""
    s1 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(s1, 1)
    fn()
    s2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(s2, 2)


def f3(x):
    return "–" if x is None else f"{x:.3f}"


# =================================================================== TITLE
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Radview: An Agentic Early-Warning System that Turns Restaurant Reviews into Verified Action Plans")
r.font.size = Pt(22)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_runs(p, "Yeo Zhang Han, Daniel Leong, Peh Xue Ying, Yang Jiajun Joven, Wong Wei Xuen, Vijayanand Jeeva", size=11)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_runs(p, "*DSA2111 Cloud-based Data Science and Programming — Group 10*", size=10)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
add_runs(p, "*National University of Singapore*  ·  Code: github.com/Chef2121/dsa2111_project", size=9)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
add_runs(p, "[[Author order to be confirmed by the group]]", size=8)

body = doc.add_section(WD_SECTION.CONTINUOUS)
set_columns(body, 2)

# =================================================================== ABSTRACT
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r = p.add_run("Abstract—")
r.bold, r.italic, r.font.size = True, True, Pt(9)
add_runs(p, (
    "Online ratings drive restaurant revenue, yet owners discover a reputation problem only after their average "
    "rating has already fallen, and the aggregate score never says what to fix. Existing research predicts permanent "
    "business failure from rating sequences, and commercial review analytics report overall sentiment; both describe "
    "the past and stop at diagnosis. We present Radview, a cloud-native system that warns owners of an impending rating "
    "decline and tells them, with cited review evidence, which operational problem to address first. Radview labels "
    "every Yelp review's sentiment toward food, service, cleanliness, price and ambience with "
    "a DistilBERT model fine-tuned by weak supervision, tracks aspect-level complaint rates with robust control charts, "
    "and forecasts a statistically significant 0.3-star decline over the next three months with a calibrated "
    "PyTorch GRU. Forecasts are explained by SHAP and a counterfactual, and a LangGraph agent turns them "
    "into a ranked action plan whose every claim is checked by a mechanical five-axis verifier before an owner sees "
    "it. The system runs as Docker containers on Kubernetes in AWS, behind a REST API and an owner dashboard. On "
    "152,649 reviews of 397 Philadelphia restaurants, the "
    f"aspect model reaches macro-F1 {res2['distilbert_finetuned']['macro_f1']:.3f} on human-labelled SemEval-2014 "
    f"sentences, ahead of TF-IDF and lexicon baselines but behind a zero-shot open LLM "
    f"({res2['zero_shot_llm']['macro_f1']:.3f}) that is ~{LLM_SLOWDOWN:.0f}× slower; the verifier rejects every claim of an "
    f"adversarial planner, including realistic wrong-figure claims, and passes only "
    f"{m5c['planners']['open_model']['groundedness']:.0%} of the claims of a free open model (Qwen 2.5 7B) acting as "
    f"the agent. On an out-of-time, COVID-era test period a rating-trend logistic regression (AUROC "
    f"{m4['results']['rating_logreg']['auroc']:.3f}) beats the GRU ({m4['results']['radview_gru']['auroc']:.3f}) and "
    "is the only model whose early warnings clearly beat random alerts; complaint spikes do not anticipate a further "
    "decline at monthly granularity but do mark lasting damage, and all conclusions hold on a survivor-free selection "
    "of restaurants."),
    size=9, bold=True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_before = Pt(4)
r = p.add_run("Index Terms—")
r.bold, r.italic, r.font.size = True, True, Pt(9)
add_runs(p, "aspect-based sentiment analysis, early warning, rating forecasting, agentic AI, LLM grounding, "
            "explainable AI, Kubernetes.", size=9, bold=True)

# =================================================================== I. INTRODUCTION
h1("Introduction")
para("A single star on Yelp is worth real money: a one-star increase in a restaurant's rating has been associated "
     "with a 5–9% increase in revenue, with the effect concentrated in independent restaurants rather than chains [1], "
     "and rounding thresholds in displayed ratings measurably change how often diners choose a venue [2]. At the same "
     "time, restaurants operate on thin margins and fail at high rates, often for operational reasons the owner could "
     "in principle have fixed [3]. Reviews therefore carry an early, free and detailed signal about what customers "
     "are unhappy with — but an independent restaurant has neither the analyst to read hundreds of reviews nor the "
     "budget for enterprise reputation software.")
para("The tools available today leave a gap in three places. First, the **star rating is a lagging, aggregate "
     "indicator**: it accumulates over years, moves slowly, and treats a three-star review that praises the food but "
     "complains about service identically to one that complains about hygiene. Second, **research on review-based "
     "risk prediction stops at diagnosis**. The strongest work we found, a variable-duration hidden Markov model of "
     "rating sequences, predicts restaurant failure with an out-of-sample AUC of 85.2% but, as its authors note, "
     "does not tell managers how to reach a better state and uses review text only as one aggregate sentiment score "
     "[4]. A hybrid CNN–DNN that adds aspect-wise topic sentiment predicts permanent closure with about 90% "
     "accuracy, but again outputs a single open/closed label at a fixed horizon [5]. Third, **systems that do "
     "generate recommendations with large language models can hallucinate**, and an owner cannot easily tell a "
     "grounded recommendation from an invented one.")
para("Radview addresses these gaps with one end-to-end pipeline. We formulate the task as follows. For restaurant "
     "*i* and calendar month *t*, let *x*_{i,≤t} be the sequence of monthly signals derived only from reviews, "
     "check-ins and tips up to and including month *t*. The target *y*_{i,t} = 1 if the mean star rating over months "
     "*t*+1…*t*+3 falls at least 0.3 stars below the mean over months *t*−5…*t* **and** the drop is statistically "
     "significant (one-sided *z* ≥ 1.645). Radview must (i) estimate P(*y*_{i,t} = 1 | *x*_{i,≤t}) "
     "without using any future data, (ii) attribute the risk to five operational aspects, and (iii) produce a "
     "short, ranked action plan in which every claim is traceable to the analytics or to a verbatim review.")
para("Our approach is to decompose reviews into aspect-level complaints with a transformer, monitor those "
     "complaints the way a quality engineer monitors a production line, forecast decline with a sequence model, and "
     "let an agent write the plan under a strict, mechanical verifier. Targeting a *rating decline* rather than "
     "closure gives owners an earlier, milder signal with room to intervene. The main contributions are:")
bullets([
    "**An aspect-to-action pipeline on real data.** Weakly supervised aspect-level sentiment read from review text "
    "(the star rating only fills in polarity for opinion-less sentences in the training labels — about 4 in 10 "
    "labels) feeds complaint monitoring, a calibrated decline forecaster with two independent "
    "explanations, and an action planner — evaluated on 152,649 real Yelp reviews and on human-labelled "
    "SemEval-2014 sentences.",
    "**A grounded agent with a five-axis verifier.** A LangGraph agent in which an LLM — Claude, or a free open-weight "
    "model served on our own GPU — chooses which analytics tools to call, with a deterministic planner as fallback "
    "and baseline; every claim is checked for citation, verbatim "
    "quotation, aspect relevance, numeric fidelity (each figure against the quantity its sentence names) and "
    "substance, and an adversarial planner with blatant and realistic fabrications demonstrates that they are rejected.",
    "**A reproducible cloud-native deployment with honest evaluation.** Docker images gated by tests, Kubernetes "
    "jobs and services, a REST API and an owner dashboard; and an evaluation against six forecasting baselines "
    "that reports where the proposal's hypotheses did *not* hold.",
])

# =================================================================== II. RELATED WORK
h1("Related Work")
h2("Online Reviews and Business Outcomes")
para("Luca [1] combined Yelp ratings with Washington State revenue data and a regression-discontinuity design to show "
     "that ratings causally affect demand for independent restaurants. Anderson and Magruder [2] used the same rounding "
     "discontinuities to show that an extra half-star raises the chance of a restaurant selling out at peak times. "
     "Parsa et al. [3] interviewed owners of failed and surviving restaurants and found that failure is usually "
     "operational and gradual rather than sudden. Together these studies motivate monitoring reviews continuously "
     "and acting early — but they are explanatory studies and do not build a predictive tool.")
h2("Predicting Decline and Failure from Reviews")
para("Naumzik et al. [4] model each restaurant's rating sequence with a variable-duration hidden Markov model and "
     "detect an at-risk latent state long before closure, achieving an out-of-sample AUC of 85.2%; text enters only "
     "as one sentiment score per review, so the model cannot say which aspect drives risk. Thazhackal and Devi [5] "
     "combine aspect-wise sentiment from a sentiment-aligned topic model with business attributes in a hybrid "
     "CNN–DNN and report around 90% cross-validated accuracy for restaurant closure, but predict a single binary "
     "outcome at a fixed horizon and evaluate with a random rather than chronological split. Earlier work showed that "
     "review text improves rating prediction over stars alone: Ganu et al. [8] used sentence-level topics and "
     "sentiment in restaurant reviews, and McAuley and Leskovec [9] aligned latent rating dimensions with review "
     "topics. These methods predict a rating, not a future *change* in rating, and none produce recommendations.")
h2("Aspect-Based Sentiment Analysis")
para("Hu and Liu [6] introduced feature-based opinion mining with frequent nouns and opinion lexicons — the "
     "lexicon-plus-rules approach our weak labeller descends from. The SemEval-2014 Task 4 benchmark [7] "
     "standardised aspect-category sentiment for restaurant reviews with human annotations for food, service, price, "
     "ambience and a miscellaneous category; we use its test set as an external, human-labelled evaluation. "
     "Transformer models dominate the task: BERT [10] fine-tuned with auxiliary sentences [12] or with review-domain "
     "post-training [13] gives large gains when trained on SemEval's own labels. DistilBERT [11] retains most of "
     "BERT's accuracy at 40% fewer parameters, which matters for CPU-only serving. Because Yelp has no aspect labels, "
     "we follow the weak-supervision paradigm popularised by Snorkel [14], in which noisy heuristic labels train a "
     "model that generalises beyond the heuristics; unlike [12], [13], our model is never trained on SemEval labels — "
     "800 sentences of the SemEval training file serve only to choose between our two trained models.")
h2("Monitoring, Forecasting and Evaluation")
para("Exponentially weighted moving-average control charts [15] are a standard tool for detecting gradual drift in "
     "process quality; we apply them to complaint rates, and replace the mean/standard-deviation scale with the median "
     "absolute deviation, which Leys et al. [16] recommend for outlier detection under skew. For forecasting from "
     "monthly sequences we use gated recurrent units [17], and gradient-boosted trees in the LightGBM implementation "
     "[18] as a strong tabular baseline. Because declines are rare, we follow Saito and Rehmsmeier [19] in treating "
     "precision–recall rather than ROC analysis as the primary view, report the Brier score [20] for probability "
     "quality, calibrate scores with Platt scaling [21], and guard against temporal leakage, a pervasive source of "
     "over-optimistic results [22].")
h2("Explainability")
para("SHAP [23] attributes a prediction to its inputs using Shapley values; for differentiable models, the expected "
     "gradients estimator [24] approximates SHAP by integrating gradients from background samples. Counterfactual "
     "explanations [25] instead ask how the input would have to change for the outcome to change. We report both and "
     "measure how often they agree on the most important aspect.")
h2("LLM Agents and Grounded Generation")
para("ReAct [26] interleaves reasoning with tool calls so that a language model can gather information before "
     "answering, and retrieval-augmented generation [27] conditions generation on retrieved documents. Both reduce, "
     "but do not eliminate, hallucination, which Ji et al. [28] survey as a central risk of natural-language "
     "generation. Gao et al. [29] evaluate whether generated statements are supported by their cited passages. "
     "Radview's verifier adopts that idea in a deliberately mechanical form: it checks citations, quotations and "
     "numbers against the agent's own tool outputs rather than asking another model to judge them.")
h2("Cloud-Native Deployment and Summary of Gaps")
para("Container orchestration lets data pipelines and services be scheduled, restarted and scaled declaratively "
     "[30]. Table I summarises how Radview differs from the closest work: it predicts an earlier, milder outcome, "
     "decomposes risk into aspects using text-based polarity, explains its forecasts two ways, and converts them into "
     "verified actions inside a deployable system.")
table("Comparison with closest related systems",
      ["", "[4]", "[5]", "Review dashboards", "Radview"],
      [["Outcome predicted", "Closure", "Closure", "None", "Rating decline"],
       ["Aspect decomposition", "No", "Topics", "Sometimes", "5 aspects"],
       ["Aspect polarity from text", "No", "Yes", "Varies", "Mostly (DistilBERT)"],
       ["Leak-free time split", "Yes", "No", "n/a", "Yes"],
       ["Explanation of risk", "Latent state", "No", "No", "SHAP + counterfactual"],
       ["Actionable plan", "No", "No", "No", "Yes, verified"],
       ["Deployed system", "No", "No", "Yes", "Docker + Kubernetes"]],
      [1.0, 0.48, 0.48, 0.62, 0.82], size=7)

# =================================================================== III. SYSTEM DESIGN
h1("Proposed System Design")
h2("Overview")
para("Fig. 1 shows Radview's six modules. Raw Yelp files enter M1, which writes cleaned columnar tables to a shared "
     "volume. M2 labels every review with aspect-level sentiment; M3 aggregates those labels into monthly complaint "
     "rates and flags spikes; M4 forecasts decline risk and explains it; M5 turns the forecast, the trends and the "
     "underlying reviews into a verified action plan; and M6 serves everything through a REST API and a dashboard. "
     "Each module is a plain Python package that reads the previous module's output, so the same code runs in a "
     "notebook, in a container, or as a Kubernetes job, and a weekly job re-ingests and re-scores without retraining.")


def _fig1():
    figure(FIGS / "fig1_architecture.png",
           "System architecture. Arrows show data flow; all modules run as containers built from one image and share a "
           "persistent volume. SemEval-2014 is used only for evaluating M2.", width=Inches(6.9))


wide(_fig1)

h2("M1 — Data Ingestion and Storage")
para("The Yelp Open Dataset review file is 5.3 GB of JSON lines (6,990,280 reviews). M1 never loads it whole: it "
     "first selects restaurants from the business file (categories containing *Restaurants*), restricts to one market "
     "— the 400 most-reviewed restaurants in Philadelphia, the city with the most restaurant reviews — and then "
     "streams the review file, using a regular-expression pre-filter on the business identifier before paying for "
     "JSON parsing. Text is normalised, empty or invalid records removed, duplicates dropped both by review ID and by "
     "(user, business, text), and the final partial calendar month discarded because a half-observed month would "
     "look like a collapse in review volume. Check-ins and tips are aggregated to business-months. Outputs are "
     "Parquet tables plus a data-quality report. A preflight command summarises restaurants and reviews per city "
     "from the business file in seconds, so a user can choose a slice before ingesting. Selecting on the *final* review "
     "count favours restaurants that survived (82% of the slice is still open, against 60% of all Philadelphia "
     "restaurants), so a second slice — the 400 restaurants with the most reviews in 2016, using no information from "
     "after the start of the study — serves as a robustness check (Section V). Each restaurant's cuisine, used to pick "
     "comparable peers, is its most common food category, skipping venue and drinks tags such as *Nightlife* or *Bars*. "
     "Table II gives the resulting dataset; a Yelp-format simulator with injected deterioration episodes provides a "
     "controlled third dataset.")
table("Dataset after ingestion (Yelp, Philadelphia)",
      ["Quantity", "Value"],
      [["Restaurants (top 400 by reviews; with 2016+ reviews)", "397"],
       ["Reviews streamed / kept for these restaurants", "6,990,280 / 312,317"],
       ["Reviews after date filter, cleaning, de-duplication", "152,649"],
       ["Period", "Jan 2016 – Dec 2021"],
       ["Mean star rating / median review length", "4.01 / 75 words"],
       ["Business-months with check-ins / tips", "44,515 / 20,232"]],
      [2.0, 1.4])

h2("M2 — Aspect-Level Sentiment Classification")
para("Each review receives one of three labels — *not mentioned*, *positive* or *negative* — for each of five "
     "aspects. Because Yelp provides no aspect labels, we generate them by weak supervision. Each sentence is "
     "assigned to the aspects whose seed terms it contains (e.g. *waiter*, *waited* → service; *sticky*, *restroom* → "
     "cleanliness); its polarity comes from positive and negative opinion words with a three-token negation window "
     "(*not clean* is a complaint, while *nothing but great* is not), and fixed phrases such as *can't wait to come "
     "back* or *worth the wait* are ignored. Only when a sentence carries no opinion word do we fall back on the "
     "review's star rating (4–5★ positive, 1–2★ negative; a mixed 3★ review gives no label) — which still supplies "
     "about 4 in 10 labels, so the trained models learn partly from stars. Generic words such as *ordered* or *dinner* "
     "are excluded from the seed lists after we found they copied the star rating onto neutral sentences. On 40,000 "
     "weakly labelled reviews, food is mentioned in most reviews and cleanliness in only a few percent, so the label "
     "distribution is highly imbalanced.")
para("Three models are trained on these labels. The *lexicon rule* is the labeller itself. The *TF-IDF baseline* "
     "fits one class-balanced logistic regression per aspect on word uni- and bigrams. The proposed model fine-tunes "
     "DistilBERT [11]: the [CLS] representation feeds a linear layer with 5 × 3 outputs, one softmax head per aspect, "
     "trained with class-weighted cross-entropy (weights ∝ inverse class frequency^0.5) for two epochs on 6,000 "
     "reviews (sequence length 256 — 128 tokens cut off about 40% of Yelp reviews — AdamW, learning rate 3×10⁻⁵, "
     "warm-up and linear decay). For serving on CPU the "
     "encoder's linear layers are dynamically quantised to int8, which cut inference time by about 30% in our "
     "measurements. The model with the best macro-F1 on a development set of 800 sentences from the SemEval-2014 "
     "training file is selected automatically and scores every review; the SemEval test set is used only for reporting. "
     "As an untrained fourth baseline, a zero-shot LLM — the same open Qwen 2.5 7B model the agent uses, served by "
     "Ollama on a T4 GPU — is prompted to label each text directly and answer with a JSON object; it is "
     "evaluated but not used for scoring, because it needs a GPU and is far slower.")

h2("M3 — Emerging Complaint Detection")
para("For restaurant *i*, aspect *a* and month *t*, the complaint rate is the share of reviews labelled negative on "
     "*a*. Monthly counts are sparse, so we smooth the numerator and denominator separately with an EWMA (span 3) "
     "and take their ratio, then compute a robust z-score against the restaurant's *own* previous twelve months:")
para("*z*_{t} = (*r*_{t} − *m*_{t}) / max(1.4826·MAD_{t}, binomial s.e., 0.04),  *m*_{t} = median(*r*_{t−12}, …, *r*_{t−1}),",
     indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
para("where MAD_{t} is the median absolute deviation of the same twelve months from *m*_{t}. "
     "The floor stops a perfectly stable history from producing an infinite score, and the binomial term widens the "
     "scale for low-volume restaurants. An aspect is flagged when *z* > 2.5 and at least two negative mentions occurred "
     "in the last two months, so that \"100% of two reviews\" cannot fire an alert. For comparison, the same detector "
     "is applied to the smoothed star rating itself (a *rating alarm*), with the sampling error of a mean rating "
     "(star standard deviation / √*n*) in place of the binomial term. Every M3 result is reported next to a *random "
     "timing* baseline: the same alerts moved to random months of the same restaurant.")

h2("M4 — Rating-Decline Forecasting")
para("M4 builds one row per restaurant-month (including months without reviews, so windows are true calendar "
     "windows) with 47 features in three groups: **rating trend and volume** (1-, 3- and 6-month and cumulative "
     "ratings, slope, review velocity, check-ins, tips, months active, and the 3-month rating minus the city median "
     "that month); **overall sentiment** (share of low-star reviews and of reviews with any complaint); and **aspect "
     "signals** for each of the five aspects (complaint rate, its change, its gap to the market median, praise rate, "
     "M3 z-score and recent spike). Every feature uses data up to month *t* only; a unit test verifies that changing "
     "future months changes the label but no feature. The label requires both a 0.3-star drop and a significant "
     "one-sided z-test, because without the test small samples produced many \"declines\" that were regression to the "
     "mean after a lucky month. Formally, with *n* reviews, mean rating μ and variance *s*² in the past "
     "window *p* (months *t*−5…*t*) and the future window *f* (months *t*+1…*t*+3):")
para("*y*_{i,t} = 1  ⇔  μ_{p} − μ_{f} ≥ 0.3  and  (μ_{p} − μ_{f}) / √(*s*_{p}²/*n*_{p} + *s*_{f}²/*n*_{f}) ≥ 1.645,",
     indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
para("At least six past and three future reviews are required; otherwise the month is left unlabelled.", indent=False)
para("The proposed forecaster is a GRU [17] that reads the last twelve months of standardised features (left-padded "
     "with a padding flag): a 64-unit projection with GELU and dropout, a 64-unit GRU, and a small head on the final "
     "hidden state. Three networks with different seeds are trained with class-weighted binary cross-entropy and "
     "early stopping on validation PR-AUC, averaged, and Platt-calibrated on the validation months [21]. Risk tiers "
     "are defined as *lift* over the typical restaurant — the mean predicted risk of the restaurants active that month "
     "(≥1.5× medium, ≥2× high) — because calibrated probabilities for a 6% event rarely exceed 20%. Forecasts are "
     "explained in two ways: SHAP values from the expected-gradients "
     "estimator [23], [24], summed over time and grouped by aspect, and a counterfactual that resets one aspect's "
     "features to the market median for the whole window and re-runs the network [25]. A ridge regression of monthly "
     "mean stars on aspect complaint rates, constrained so that a complaint can only cost stars, converts \"how far "
     "above the market\" into an estimated star cost.")

h2("M5 — Grounded Agentic Prioritisation")
para("M5 is a LangGraph state machine (Fig. 2). It first calls four tools — *query_metrics*, *run_forecast*, "
     "*get_evidence_reviews* and *compare_peers* — each a thin, read-only function over the M2–M4 outputs scoped to "
     "one restaurant. Aspects are ranked by priority = (0.7·impact + 0.3·urgency) / effort, where impact averages each "
     "aspect's share of all risk-raising SHAP (non-aspect drivers such as the rating trend included) and the share of "
     "the risk its counterfactual removes, urgency combines the M3 z-score and complaint "
     "growth, and effort is a stated assumption (cleanliness fixes are cheap; ambience needs capital). Evidence "
     "retrieval re-scores every sentence of candidate reviews with M2 and keeps a review only if its most negative "
     "sentence *on its own* is classified as a complaint about that aspect.")


figure(FIGS / "fig2_agent.png",
       "M5 agent workflow. An LLM planner (Claude or the open model) may redraft once when claims fail verification; unverified claims never "
       "reach the owner but are logged.")
para("Three planners sit behind the *draft* node. The **Claude planner** receives the pre-ranked priorities and the tool "
     "definitions and runs its own tool-calling loop — for instance fetching evidence for an aspect the analytics did "
     "not rank first — then submits the plan through a strict JSON-schema tool; every evidence ID it sees is recorded. "
     "The **open-model planner** runs the same prompt, tools and loop against a free open-weight model (Qwen 2.5 7B) "
     "served by Ollama through an OpenAI-compatible API, so it needs no API key, has no per-token cost and keeps "
     "reviews on the group's own server; it repairs only output-format slips (e.g. tool arguments sent as strings) and, "
     "when greedy decoding falls into a repetition loop that the server aborts, retries that turn once at temperature "
     "0.4. The **rule-based planner** fills templates from the same tool outputs and the action playbook; it is both the "
     "offline fallback and the baseline. The **verifier** then checks every claim on five axes: *citation* (the review "
     "exists, belongs to this restaurant and was retrieved in this run), *quotation* (a quote is given and appears "
     "verbatim), *relevance* (M2 labels the review negative on the aspect and the quoted sentence alone is a complaint "
     "about it), *statistic* (every percentage is within two points, and every star figure within 0.05, of the tool "
     "output its own clause refers to — the market median only where the clause mentions the market, the risk only "
     "next to *risk* or *chance* — and multipliers such as *tripled* are rejected as uncheckable) and *substance* (a "
     "metric claim must state a checkable figure). An LLM's summary and headlines are checked for unsupported figures "
     "too. Failing claims are removed, and the share that passes is the plan's groundedness. A hallucinating planner "
     "that fabricates one claim per failure mode, plus five realistic mistakes built from real figures and real "
     "citations, is part of the test suite.")

h2("M6 — Serving, Containers and Orchestration")
para("A FastAPI service exposes ranked restaurants, per-restaurant metrics and timelines, action plans, live alerts, "
     "on-the-fly analysis of raw review text, evaluation results, and an endpoint for owners to rate a plan's "
     "usefulness. The Streamlit dashboard (Fig. 3) follows the order of an owner's questions — *am I in trouble*, *is "
     "my rating moving*, *what are people complaining about compared with the market*, *is it getting worse*, and "
     "*what do I do first* — and every chart has a table view and light and dark palettes. One CPU-only Docker image "
     "serves all three roles; its build includes a test stage, so an image cannot be produced from code that fails "
     "its tests. Kubernetes manifests define a namespace, a persistent volume claim, a training Job, a weekly "
     "re-scoring CronJob and health-checked Deployments for the API and the dashboard (two replicas). The system is "
     "deployed on the group's cloud machine, an AWS EC2 t3.large (2 vCPU, 8 GB) provisioned through NUS RONIN in "
     "the Singapore region: a single-node k3s Kubernetes cluster runs the API Deployment and the two dashboard "
     "replicas from the same manifests, using the host's Docker runtime so the test-gated image is not stored "
     "twice, with the trained models on a persistent volume. The dashboard Service is reachable only through an "
     "SSH tunnel, because it has no login and serves Yelp review text; the pods restart automatically whenever "
     "the machine is started, so it can be stopped between uses to stay within the $100 budget. The training "
     "Job is not run there: models are trained once and shipped as a bundle. For the open-model experiment the "
     "same machine was temporarily switched to a g4dn.xlarge with one NVIDIA T4 GPU running Ollama. One command "
     "(scripts/deploy_ronin_k3s.sh) reproduces the deployment. Table III lists the main frameworks.")
figure(FIGS / "fig5_dashboard.png",
       "Owner dashboard on real data: plain-English verdict, KPI tiles, monthly rating (gaps where no reviews), complaint "
       "rates against the market median (ticks), and the top of the verified action plan with its small-sample caveat.")
table("Implementation stack",
      ["Layer", "Technology"],
      [["Data processing", "pandas, PyArrow (Parquet)"],
       ["Aspect models", "PyTorch, Hugging Face Transformers, scikit-learn"],
       ["Forecasting", "PyTorch (GRU), LightGBM, scikit-learn"],
       ["Explanation", "SHAP (Gradient & Tree explainers)"],
       ["Agent", "LangGraph, Anthropic SDK (Claude), Ollama (Qwen 2.5 7B)"],
       ["Serving", "FastAPI, Uvicorn, Streamlit, Altair"],
       ["Deployment", "Docker (multi-stage), Kubernetes (k3s), AWS EC2 via NUS RONIN, pytest"]],
      [1.0, 2.4])
para("Table IV lists the REST endpoints. They are thin wrappers over one read-side service layer that the dashboard "
     "also calls in-process when no API is configured, so the two can never show different numbers. The service keeps "
     "the pipeline outputs in memory and reloads them automatically when a newer pipeline run has finished, so the "
     "weekly re-scoring job reaches a long-running API without a restart. Inputs are validated at the boundary "
     "(known restaurant, known aspect, at most 200 texts per analysis call, usefulness between 1 and 5), and a plan "
     "refresh requested through the API uses the rule-based planner unless the operator explicitly allows LLM calls, "
     "because an LLM run costs money and the dashboard has no login.")
table("REST API of the serving layer (M6)",
      ["Endpoint", "Returns"],
      [["GET /restaurants", "All monitored restaurants ranked by decline risk"],
       ["GET /restaurants/{id}", "Metrics vs. market, forecast with SHAP, counterfactual and star cost, peers"],
       ["GET /restaurants/{id}/timeline", "Monthly ratings, complaint rates vs. market, spikes, risk"],
       ["GET /restaurants/{id}/reviews", "Review explorer, filtered by aspect and sentiment"],
       ["GET /restaurants/{id}/plan", "Verified action plan with tool calls and rejected claims"],
       ["GET /alerts", "This month's complaint spikes, most severe first"],
       ["POST /analyse", "Aspect sentiment and complaint sentence for raw review texts"],
       ["GET /evaluation", "M2–M5 evaluation results and M4 PR/ROC curves"],
       ["POST /feedback", "Stores an owner's 1–5 usefulness rating of a plan"]],
      [1.35, 2.05], size=7)
para("Table V shows how the same image is orchestrated. A one-off Job trains and evaluates every model, a weekly "
     "CronJob ingests new reviews and re-scores them with the saved models (no retraining, and no paid LLM "
     "comparison runs), and two Deployments serve the API and the dashboard from a shared persistent volume, with "
     "readiness and liveness probes so that Kubernetes restarts a stuck pod. An optional Deployment runs the "
     "open-weight LLM server for the agent.")
table("Kubernetes objects of the deployment",
      ["Object", "Role"],
      [["Job radview-train", "Full pipeline: ingest, train and evaluate every model, plans"],
       ["CronJob radview-rescore", "Mondays 03:00: ingest, re-score with saved models, regenerate plans"],
       ["Deployment radview-api", "FastAPI with readiness / liveness probes"],
       ["Deployment radview-dashboard", "Streamlit, two replicas, talks to the API"],
       ["PersistentVolumeClaim radview-data", "Shared data, models and outputs"],
       ["Deployment ollama (optional)", "Open-weight LLM server (Qwen 2.5 7B) for the agent"]],
      [1.45, 1.95], size=7)

h2("Implementation Quality and Testing")
para("Each module is a Python package with a single entry point and a docstring that states its inputs, outputs and "
     "design decisions, and every tunable — thresholds, windows, model names, file locations — lives in one "
     "configuration module that environment variables override, so the same code runs on a laptop, in Docker and "
     "under Kubernetes. Correctness is checked by an automated suite of 43 tests (Table VI), which the Docker build "
     "runs as a stage: an image cannot be built from code that fails its tests. The tests target the properties the "
     "results depend on rather than lines of code: that features at month *t* never change when later months change; "
     "that the time split leaves a three-month gap before each boundary; that the M3 baseline is the true median and "
     "median absolute deviation of past months; that the verifier rejects real figures attached to the wrong "
     "quantity, empty quotations and uncheckable multipliers while accepting the rule-based wording; and that the "
     "LLM planners fall back to the rule-based planner when no model is reachable. A code review against the real "
     "outputs found and fixed several such issues before the final run, including a temporal leak at the split "
     "boundaries, model selection on the test set and a verifier that accepted any matching number; Section IV "
     "reports the results after these fixes.")
table("Automated test suite",
      ["Area", "Tests", "Examples of what is checked"],
      [["M1 ingestion", "3", "Restaurant filter, de-duplication, cuisine, first-year selection"],
       ["M2 aspects", "7", "Negation, idioms, star fallback, zero-shot parsing"],
       ["M3 trends", "3", "No alert before a spike, exact MAD, chance baseline"],
       ["M4 forecast", "8", "No future leakage, purged split, padding, calibration, star costs"],
       ["M5 agent", "14", "Verifier axes, adversarial plans, LLM tool loop and fallbacks"],
       ["M6 serving", "8", "Endpoints, validation, 404s, reload after a new run"]],
      [0.85, 0.4, 2.15], size=7)
para("Table VII collects the main parameters of the pipeline. They are set once in the configuration module, "
     "documented next to their value, and each can be overridden by an environment variable, which is how the "
     "robustness runs in Section IV change the restaurant selection or the test window without touching the code.")
table("Key parameters of the pipeline",
      ["Module", "Parameter", "Value"],
      [["M1", "Restaurants / city / start date", "400 / Philadelphia / Jan 2016"],
       ["M2", "DistilBERT input / epochs / training reviews", "256 tokens / 2 / 6,000"],
       ["M2", "TF-IDF n-grams / training reviews", "1–2 / 40,000"],
       ["M3", "EWMA span / baseline window / minimum history", "3 / 12 / 6 months"],
       ["M3", "Spike threshold / minimum recent complaints", "z > 2.5 / 2"],
       ["M4", "Decline: drop / significance / horizon", "≥ 0.3★ / z ≥ 1.645 / 3 months"],
       ["M4", "Minimum reviews past / future window", "6 / 3"],
       ["M4", "Test / validation / purge gap", "18 / 6 / 3 months"],
       ["M4", "GRU window / hidden size / ensemble", "12 months / 64 / 3 seeds"],
       ["M4", "Risk tiers (lift over typical risk)", "≥ 1.5× medium, ≥ 2× high"],
       ["M5", "Evidence per aspect / look-back", "3 reviews / 4 months"],
       ["M5", "Verifier tolerance (rates / stars)", "± 2 points / ± 0.05★"],
       ["Eval.", "Early-warning budget / look-back", "15% / 6 months"]],
      [0.45, 1.75, 1.2], size=7)

# =================================================================== IV. EXPERIMENTS
h1("Experiment Evaluations and Performance Comparisons")
h2("Experimental Setup")
sp = m4["split"]
para(f"Models were trained and evaluated on one NVIDIA T4 GPU (AWS g4dn.xlarge via NUS RONIN); the deployed system "
     f"serves them on CPU. Forecasting uses a strict chronological split: "
     f"training months {sp['train'].replace('..', '–')}, validation {sp['valid'].replace('..', '–')} and test "
     f"{sp['test'].replace('..', '–')} ({sp['n_train']:,}, {sp['n_valid']:,} and {sp['n_test']:,} labelled "
     f"restaurant-months; {sp['pos_rate_train']:.1%} and {sp['pos_rate_test']:.1%} positive in training and test). "
     f"The {sp.get('gap_months', 3)} labelled months before each boundary are dropped, because their three-month label "
     f"window reaches into the next period. "
     f"The test window coincides with the COVID-19 pandemic, a regime the models never saw in training. Metrics follow "
     f"Section II: macro-F1 and complaint-F1 for M2; precision, lift and lead time for M3; AUROC, PR-AUC, Brier score, "
     f"precision and recall in the riskiest 10% of restaurant-months, and early warning at a fixed 15% alert budget "
     f"for M4; and groundedness for M5. Alert-based results are reported next to a chance level — the same alerts at "
     f"random times — because a detector that fires often warns of many declines by luck. Fixed random seeds make all "
     f"reported numbers reproducible.")

h2("Aspect Classification (M2)")
res = {r["model"]: r for r in m2["results"]["semeval2014"] if "macro_f1" in r}
lab = {"lexicon_rule": "Lexicon rule (weak labeller)", "tfidf_logreg": "TF-IDF + logistic regression",
       "distilbert_finetuned": "**DistilBERT, fine-tuned (ours)**", "zero_shot_llm": "Zero-shot Qwen 2.5 7B (GPU)"}
zs = res.get("zero_shot_llm")
table("Aspect sentiment on SemEval-2014 restaurant test set",
      ["Model", "Macro-F1", "Compl.-F1", "ms/sent."],
      [[lab[k], f3(res[k]["macro_f1"]), f3(res[k]["complaint_f1"]), f"{res[k]['inference_ms_per_review']:.2f}"]
       for k in ("lexicon_rule", "tfidf_logreg", "distilbert_finetuned", "zero_shot_llm") if k in res],
      [1.75, 0.55, 0.6, 0.5],
      note="800 human-labelled sentences; food, service, price, ambience (SemEval has no cleanliness category). "
           "No model was trained on SemEval; the production model was chosen on 800 sentences of the SemEval training "
           "file. The LLM may have seen the public benchmark during pre-training. ms/sent.: DistilBERT and the LLM on "
           "one NVIDIA T4 GPU, the lexicon rule and TF-IDF on CPU.")
tf, db = res["tfidf_logreg"], res["distilbert_finetuned"]
lx = res["lexicon_rule"]
tf_better = [a for a in ("food", "service", "price", "ambience")
             if tf["macro_f1_per_aspect"][a] > db["macro_f1_per_aspect"][a]]
para(f"Table VIII and Fig. 4 compare the aspect models on 800 human-labelled SemEval-2014 sentences that none of "
     f"them was trained on. Both learned models improve substantially on the lexicon rule that produced their training "
     f"labels (macro-F1 {f3(db['macro_f1'])} and {f3(tf['macro_f1'])} versus {f3(lx['macro_f1'])}), which confirms "
     f"that weak supervision generalises beyond its own heuristics. DistilBERT is the most accurate trained model, and "
     f"its advantage is concentrated where Radview needs it: complaint-F1 rises from {f3(tf['complaint_f1'])} to "
     f"{f3(db['complaint_f1'])} (+{db['complaint_f1'] / tf['complaint_f1'] - 1:.0%} relative), with the largest gains "
     f"on service ({f3(tf['complaint_f1_per_aspect']['service'])} → {f3(db['complaint_f1_per_aspect']['service'])}), "
     f"price ({f3(tf['complaint_f1_per_aspect']['price'])} → {f3(db['complaint_f1_per_aspect']['price'])}) and "
     f"ambience ({f3(tf['complaint_f1_per_aspect']['ambience'])} → {f3(db['complaint_f1_per_aspect']['ambience'])})."
     + (f" TF-IDF remains marginally better on {', '.join(tf_better)} macro-F1." if tf_better else "")
     + " Ambience is hardest for every model: it is mentioned in a minority of reviews and its complaints are "
     "lexically diverse. Fully supervised BERT models trained on SemEval's own training set report higher scores "
     "[12], [13]; ours is a transfer result from noisy Yelp labels.")
if zs:
    para(f"The zero-shot LLM, given no training at all, is clearly more accurate: macro-F1 {f3(zs['macro_f1'])} and "
         f"complaint-F1 {f3(zs['complaint_f1'])}, ahead of DistilBERT on every aspect and most of all at detecting "
         f"complaints (food {f3(zs['complaint_f1_per_aspect']['food'])} vs "
         f"{f3(db['complaint_f1_per_aspect']['food'])}, ambience {f3(zs['complaint_f1_per_aspect']['ambience'])} vs "
         f"{f3(db['complaint_f1_per_aspect']['ambience'])}); all {zs['n_test']} replies were valid JSON. The price is "
         f"speed: {zs['inference_ms_per_review']:.0f} ms per sentence against {db['inference_ms_per_review']:.1f} ms for "
         f"DistilBERT on the same T4 GPU (~{LLM_SLOWDOWN:.0f}×), so labelling all 152,649 reviews once would take well "
         f"over a day of GPU time instead of about 14 minutes (Sec. IV-I). The "
         f"comparison may also flatter the LLM, since SemEval-2014 is public and could be in its pre-training data. "
         f"We therefore keep DistilBERT for scoring and read the gap as a target: replacing the lexicon weak labels with "
         f"LLM labels on a sample of Yelp reviews (distillation) is the most direct way to improve M2.")
figure(FIGS / "fig3_m2.png",
       "Per-aspect F1 on SemEval-2014 for the lexicon rule, TF-IDF, fine-tuned DistilBERT and the zero-shot Qwen 2.5 7B "
       "LLM. Left: three-class macro-F1. Right: F1 for detecting complaints (negative class).")

h2("Emerging Complaint Detection (M3)")
cs, ra = m3["complaint_spike"], m3["rating_alarm"]
table("Do alerts anticipate a decline in the next three months?",
      ["Detector", "Flagged", "Lift", "Onsets warned", "Lasting-drop lift", "Lead (mo)"],
      [[n, f"{d['flag_rate']:.1%}", f"{d['lift']:.2f} ({d['chance']['lift']:.2f})",
        f"{d['onsets_detected_pct']}% ({d['chance']['onsets_detected_pct']}%)",
        f"{d['lasting_drop']['lift']:.2f} ({d['chance']['lasting_drop_lift']:.2f})",
        f"{d['median_lead_months']:g}"] for n, d in (("Complaint spike", cs), ("Rating alarm", ra))],
      [0.82, 0.42, 0.58, 0.74, 0.62, 0.4], size=7,
      note=f"Base rate {m3['base_rate']:.1%}; {cs['onsets']} decline onsets; lift = precision / base rate. In brackets: "
           f"the same alerts at random months of the same restaurant (chance level). Lasting drop = the next three months "
           f"are ≥0.3★ below the six-month rating from three months earlier (base rate "
           f"{cs['lasting_drop']['base_rate']:.0%}).")
para(f"Table IX and Fig. 5 test the proposal's central hypothesis: that aspect complaint spikes are a *leading* indicator. On "
     f"real data they are not. Restaurant-months with a complaint spike are followed by a decline only "
     f"{cs['precision']:.1%} of the time, below the {m3['base_rate']:.1%} base rate (lift {cs['lift']:.2f}) and below "
     f"what the same spikes reach at random times (lift {cs['chance']['lift']:.2f}); only "
     f"{cs['onsets_detected_pct']}% of decline onsets are preceded by a spike in the previous six months, against "
     f"{cs['chance']['onsets_detected_pct']}% by chance. The rating alarm is worse still (lift {ra['lift']:.2f}). The "
     f"reason is regression to the mean: a spike typically arrives when the rating has *already* dropped, and because "
     f"the label compares the next three months with a past window that contains the drop, the months after a spike "
     f"tend to recover rather than fall further. Measured against the rating from *before* the alert, the picture "
     f"reverses: after a complaint spike the next three months are at least 0.3★ below that earlier rating "
     f"{cs['lasting_drop']['precision']:.0%} of the time, a lift of {cs['lasting_drop']['lift']:.2f} against "
     f"{cs['chance']['lasting_drop_lift']:.2f} by chance, so spikes mark damage that lasts. M3 is therefore useful as a "
     f"*diagnostic* — it tells the owner which aspect broke, and that the damage is real — rather than as a forecaster, "
     f"which is how the agent uses it. Fig. 5 shows both targets side by side for both restaurant selections "
     f"(Section IV-F): for the decline label every observed bar sits below its random-timing bar, while for the "
     f"lasting-drop target the observed bars clearly exceed it — most of all for the rating alarm, which fires only "
     f"once damage is already visible in the stars.")
figure(FIGS / "fig6_m3_lift.png",
       "Lift of M3 alerts over the base rate for two targets. Solid bars: observed; light bars: the same alerts at random "
       "months of the same restaurant. Dotted line: no information (lift 1). Left: decline in the next three months "
       "(the forecasting label). Right: next three months at least 0.3★ below the rating from before the alert.")

h2("Rating-Decline Forecasting (M4)")
names = {"majority": "Majority (base rate)", "current_negativity": "Current negativity (no model)",
         "rating_logreg": "Rating trend, LogReg", "rating_lgbm": "Rating trend, LightGBM",
         "sentiment_lgbm": "+ overall sentiment, LightGBM", "radview_lgbm": "All features, LightGBM",
         "radview_gru": "**All features, GRU (ours)**"}
rows = []
for k, r in m4["results"].items():
    ew = r.get("early_warning") or {}
    rows.append([names[k], f3(r["auroc"]), f3(r["pr_auc"]), f"{r['brier']:.4f}", f3(r["precision_at_10pct"]),
                 f"{ew['detected_pct']}% ({ew['chance_detected_pct']}%)" if ew.get("detected_pct") is not None else "–"])
table("Decline forecasting on the out-of-time test period",
      ["Model", "AUROC", "PR-AUC", "Brier", "P@10%", "Warned"], rows, [0.98, 0.46, 0.44, 0.44, 0.44, 0.68], size=7,
      note="All models Platt-calibrated on validation. P@10% = precision in the riskiest 10% of restaurant-months; "
           "Warned = decline onsets flagged within 6 months at a 15% alert budget (in brackets: the same number of "
           "alerts at random restaurant-months).")
g, lr = m4["results"]["radview_gru"], m4["results"]["rating_logreg"]
lg = {k: m4["results"][k] for k in ("rating_lgbm", "sentiment_lgbm", "radview_lgbm")}
gw, lw = g["early_warning"], lr["early_warning"]
para(f"Table X and Fig. 6 compare seven forecasters, from a constant base rate to the full GRU. Every learned model "
     f"beats the trivial baselines. Among the models that use all signals the GRU has the best PR-AUC "
     f"({g['pr_auc']:.3f}, against at most {max(v['pr_auc'] for v in lg.values()):.3f} for LightGBM) but not the best "
     f"AUROC ({g['auroc']:.3f}, against up to {max(v['auroc'] for v in lg.values()):.3f}), with precision "
     f"{g['precision_at_10pct']:.1%} in its riskiest decile — about {g['precision_at_10pct'] / sp['pos_rate_test']:.1f} "
     f"times the {sp['pos_rate_test']:.1%} base rate. The simplest learned model, a logistic regression on rating-trend "
     f"features, is better on every ranking metric (AUROC {lr['auroc']:.3f}, PR-AUC {lr['pr_auc']:.3f}, precision "
     f"{lr['precision_at_10pct']:.1%} in its riskiest decile), and it is the only model whose early warnings clearly "
     f"beat chance: it warns of {lw['detected_pct']}% of decline onsets, against {lw['chance_detected_pct']}% for the "
     f"same number of random alerts (+{lw['detected_pct'] - lw['chance_detected_pct']:.0f} points; at most "
     f"+{max(r['early_warning']['detected_pct'] - r['early_warning']['chance_detected_pct'] for k, r in m4['results'].items() if k not in ('majority', 'rating_logreg')):.0f} "
     f"for any other model), while the GRU's {gw['detected_pct']}% is barely above its chance level of "
     f"{gw['chance_detected_pct']}%. Fig. 6 shows why the logistic regression ranks best: its precision is higher at "
     f"low recall, where an owner would set an alert threshold. Adding aspect features to LightGBM does not help either (AUROC "
     f"{m4['results']['radview_lgbm']['auroc']:.3f} vs {m4['results']['rating_lgbm']['auroc']:.3f} without them). "
     f"Consistent with M3, on this data *what* customers complain about does not improve a three-month decline forecast "
     f"beyond the rating trend itself; the most important LightGBM features by mean |SHAP| are the 6-month and "
     f"cumulative ratings. The COVID-era test window — review volume fell by more than half in 2020 — penalises the "
     f"higher-capacity models most; Section IV-E therefore repeats the comparison on a pre-pandemic test window.")
figure(FIGS / "fig4_m4_pr.png",
       f"Precision–recall curves on the test period. The dotted line is the {sp['pos_rate_test']:.1%} base rate. The "
       f"rating-trend logistic regression dominates at low recall; the GRU has the best PR-AUC of the multi-signal models.")


def _deciles(col):
    dec = pd.qcut(tp[col].rank(method="first"), 10, labels=False)
    return tp.groupby(dec).agg(pred=(col, "mean"), obs=("decline", "mean"))


tp = pd.read_parquet(Y / "m4_test_predictions.parquet")
cal_g, cal_l = _deciles("radview_gru"), _deciles("rating_logreg")
para(f"Ranking is only half of what an owner sees; the other half is the probability itself. Fig. 7 groups the test "
     f"restaurant-months into deciles of predicted risk and compares the mean prediction with the share that actually "
     f"declined. The GRU is well calibrated across the range: its mean prediction is {tp['radview_gru'].mean():.1%} "
     f"against an observed {tp['decline'].mean():.1%}, and in its riskiest decile it predicts "
     f"{cal_g['pred'].iloc[-1]:.1%} against {cal_g['obs'].iloc[-1]:.1%} observed. The logistic regression ranks better "
     f"but overstates the risk of its riskiest restaurants ({cal_l['pred'].iloc[-1]:.1%} predicted against "
     f"{cal_l['obs'].iloc[-1]:.1%} observed), so the two Brier scores are close ({lr['brier']:.4f} vs {g['brier']:.4f}) "
     f"despite the ranking gap. Because both were calibrated on validation months before the pandemic, this "
     f"agreement on the COVID-era test period is reassuring: the probabilities shown on the dashboard can be read "
     f"at face value.")
figure(FIGS / "fig7_m4_calibration.png",
       "Calibration on the test period: mean predicted risk against the observed decline rate, by decile of predicted "
       "risk. Points on the dotted diagonal are perfectly calibrated.")
ep = m4s["injected_episodes"]
syn_g, syn_lr = ep["radview_gru"], ep["rating_logreg"]
lgbm_vis = [ep[k]["before_visible_pct"] for k in ("rating_lgbm", "sentiment_lgbm", "radview_lgbm")]
agr = m4["attribution_agreement"]
sic = agr["star_impact_coefficients"]
para(f"On the simulator, where the timing of injected deterioration episodes is known, the GRU caught "
     f"{syn_g['detected_pct']}% of {syn_g['episodes']} episodes and {syn_g['before_visible_pct']}% before the drop "
     f"was visible in the three-month rating, against {syn_lr['before_visible_pct']}% for the rating-trend logistic "
     f"regression and {min(lgbm_vis)}–{max(lgbm_vis)}% for the LightGBM models; no model catches most episodes before "
     f"they show. The two explanation methods agree on the top aspect for "
     f"{agr['all']['top_aspect_agreement']:.0%} of the {agr['all']['n']} restaurants "
     f"({agr['at_risk']['top_aspect_agreement']:.0%} of the {agr['at_risk']['n']} at risk; mean Spearman correlation "
     f"{agr['all']['mean_spearman']:.2f}) — a moderate agreement, which is why the dashboard shows both. The star-impact "
     f"model, constrained so that complaints can only cost stars, assigns the largest costs to food ({sic['food']:.2f} "
     f"stars per unit complaint rate) and service ({sic['service']:.2f}), then cleanliness ({sic['cleanliness']:.2f}) "
     f"and price ({sic['price']:.2f}); ambience comes out at {sic['ambience']:.2f} — unconstrained, it had a small "
     f"positive coefficient, a correlation artefact.")

h2("Robustness: A Pre-Pandemic Test Window")
pre = m4pre["results"]
psp = m4pre["split"]
table("AUROC / PR-AUC in two test windows",
      ["Model", "2020-04 – 2021-09", "2018-07 – 2019-12"],
      [[names[k].replace("**", ""), f"{m4['results'][k]['auroc']:.3f} / {m4['results'][k]['pr_auc']:.3f}",
        f"{pre[k]['auroc']:.3f} / {pre[k]['pr_auc']:.3f}"] for k in m4["results"] if k != "majority"],
      [1.4, 1.0, 1.0], size=7,
      note=f"Pre-pandemic run: training {psp['train'].replace('..', '–')}, validation {psp['valid'].replace('..', '–')}, "
           f"{psp['n_test']:,} test restaurant-months ({psp['pos_rate_test']:.1%} positive). Evaluation only.")
para(f"Because the main test window overlaps the pandemic, we repeated the comparison with all labels after December "
     f"2019 hidden, which moves the test window to July 2018 – December 2019 and leaves two years for training "
     f"(Table XI). Absolute scores fall for every model — there is half as much training data — but the ordering is "
     f"unchanged: the rating-trend logistic regression remains best (AUROC {pre['rating_logreg']['auroc']:.3f}, "
     f"PR-AUC {pre['rating_logreg']['pr_auc']:.3f}), the GRU second ({pre['radview_gru']['auroc']:.3f}, "
     f"{pre['radview_gru']['pr_auc']:.3f}), and the LightGBM models degrade most (AUROC "
     f"{min(pre[k]['auroc'] for k in ('rating_lgbm', 'sentiment_lgbm', 'radview_lgbm')):.3f}–"
     f"{max(pre[k]['auroc'] for k in ('rating_lgbm', 'sentiment_lgbm', 'radview_lgbm')):.3f}). The advantage of the "
     f"simple model is therefore not a COVID artefact. It is better explained by sample size: with about 6% positives, "
     f"a few hundred decline events per year are too few for high-capacity models to learn interactions that a "
     f"linear model on well-chosen trend features does not already capture. The sequence model's inductive bias — "
     f"reading the trajectory rather than hand-made windows — narrows but does not close that gap. In this window "
     f"every model warns of fewer decline onsets than the same number of random alerts would, so no early-warning "
     f"ability is shown before the pandemic either.")

h2("Robustness: A Survivor-Free Restaurant Selection")
rm2 = {r["model"]: r for r in rob["m2_evaluation"]["results"]["semeval2014"] if "macro_f1" in r}
rm3, rm4, rm5 = rob["m3_evaluation"]["complaint_spike"], rob["m4_evaluation"]["results"], rob["m5_planner_comparison"]
main_rows = [
    ["Restaurants still open (2022)", "82%", "77%"],
    ["M2 DistilBERT macro-F1 / complaint-F1", f"{f3(db['macro_f1'])} / {f3(db['complaint_f1'])}",
     f"{f3(rm2['distilbert_finetuned']['macro_f1'])} / {f3(rm2['distilbert_finetuned']['complaint_f1'])}"],
    ["M3 spike lift (chance)", f"{cs['lift']:.2f} ({cs['chance']['lift']:.2f})",
     f"{rm3['lift']:.2f} ({rm3['chance']['lift']:.2f})"],
    ["M3 lasting-drop lift (chance)", f"{cs['lasting_drop']['lift']:.2f} ({cs['chance']['lasting_drop_lift']:.2f})",
     f"{rm3['lasting_drop']['lift']:.2f} ({rm3['chance']['lasting_drop_lift']:.2f})"],
    *[[f"M4 {names[k].replace('**', '')}, AUROC / PR-AUC", f"{m4['results'][k]['auroc']:.3f} / {m4['results'][k]['pr_auc']:.3f}",
       f"{rm4[k]['auroc']:.3f} / {rm4[k]['pr_auc']:.3f}"] for k in ("rating_logreg", "sentiment_lgbm", "radview_gru")],
    ["M5 adversarial claims rejected", f"{1 - m5c['planners']['adversarial']['groundedness']:.0%}",
     f"{1 - rm5['planners']['adversarial']['groundedness']:.0%}"],
]
table("Main analysis vs. a survivor-free restaurant selection",
      ["Result", "Lifetime top 400", "2016 top 400"], main_rows, [1.6, 0.9, 0.9], size=7,
      note="Both slices: Philadelphia, the same pipeline and seeds. 287 restaurants are in both; all Philadelphia "
           "restaurants: 60% still open.")
para(f"The main slice keeps the 400 restaurants with the most reviews by the end of the dataset, which uses future "
     f"information: restaurants that later declined and closed are less likely to qualify. To test whether this "
     f"matters we re-ran the whole pipeline on the 400 restaurants with the most reviews in 2016, which uses only "
     f"information available at the start of the study. The two slices share 287 restaurants; of the 113 that enter, "
     f"46 later closed. Table XII shows that every conclusion holds: DistilBERT's accuracy is within a point, "
     f"complaint spikes again fall below chance as predictors yet mark lasting damage, the rating-trend logistic "
     f"regression is again the best forecaster (AUROC {rm4['rating_logreg']['auroc']:.3f}), and the verifier again "
     f"rejects every fabricated claim. The selection biases the sample towards successful restaurants, but not the "
     f"findings.")

h2("Discussion and Threats to Validity")
para("*Construct validity.* Our decline label is a statistical construct; a restaurant owner may care about a smaller "
     "but persistent drop, or about review volume falling. The significance test makes positives more meaningful but "
     "also rarer, which limits what any model can learn. *Internal validity.* All features are computed from past data "
     "and the split is chronological (both enforced by tests), but the aspect labels come from a model trained on "
     "reviews spanning the whole period; because M2 is trained on weak labels rather than on outcomes, we judge the "
     "resulting leakage negligible, and a stricter design would retrain M2 on training-period reviews only. "
     "The validation months used for early stopping and calibration end in December 2019, so their labels reach into "
     "the first pandemic months. "
     "*External validity.* The data cover one city and its 400 busiest restaurants, which are larger and better "
     "reviewed than the typical independent restaurant Radview targets (mean 3.91★ against 3.56★ for all Philadelphia "
     "restaurants); a survivor-free selection leaves the conclusions unchanged (Table XII), but a quieter restaurant "
     "produces exactly the sparse, noisy signals that our small-sample caveat warns about. *Measurement.* SemEval "
     "sentences are shorter and cleaner than Yelp reviews and have no cleanliness category, so M2's in-domain accuracy "
     "is unknown until the exported Yelp sample is hand-labelled, and about 4 in 10 weak labels take their polarity "
     "from the star rating. Finally, the verifier proves traceability, not truth: a plan that faithfully cites a "
     "miscalibrated forecast still passes, figures written as words are not recognised, and its relevance check relies "
     "on the same M2 model that selected the evidence.")

h2("Grounded Action Plans (M5)")
pl = m5c["planners"]
PLANNER_NAMES = {"rule_based": "Rule-based", "open_model": "Qwen 2.5 7B", "claude": "Claude", "adversarial": "Adversarial"}
table("Planner comparison on the 20 riskiest restaurants",
      ["Planner", "Grounded", "Claims", "Cit.", "Quo.", "Rel.", "Stat.", "Sub."],
      [[PLANNER_NAMES.get(k, k), f"{v['groundedness']:.0%}", str(v["claims_total"]),
        *[str(v["failures_by_axis"][a]) for a in ("citation", "quotation", "relevance", "statistic", "substance")]]
       for k, v in sorted(pl.items(), key=lambda kv: list(PLANNER_NAMES).index(kv[0]) if kv[0] in PLANNER_NAMES else 9)],
      [0.72, 0.56, 0.44, 0.34, 0.34, 0.34, 0.34, 0.32], size=7,
      note="Last five columns: claims rejected on each verifier axis (a claim can fail several). Qwen 2.5 7B ran "
           "through Ollama on one NVIDIA T4 GPU (AWS g4dn.xlarge via NUS RONIN). The Claude planner could not be run: "
           "no API key was available.")
para(f"Across all {m5b['plans']} at-risk restaurants, the rule-based planner produced {m5b['claims_total']} claims and "
     f"all of them passed the verifier, with a median latency of {m5b['median_latency_s']:.2f} s per plan. This is "
     f"expected by construction — the planner only restates tool outputs — so the verifier's value is shown by the "
     f"adversarial planner (Table XIII): every one of its {pl['adversarial']['claims_total']} fabricated claims was "
     f"rejected and every axis fired, from invented review IDs (citation) to quoted praise passed off as complaints "
     f"(relevance) and, among the five realistic mistakes, real figures attached to the wrong quantity (statistic). "
     f"Inspecting real plans exposed issues we then fixed: rates such as \"33% of reviews\" can mean one of three "
     f"reviews in the sparse late-2021 data, so plans now state counts and add a small-sample caveat below ten reviews; "
     f"and an earlier version of the verifier accepted a percentage that matched *any* returned figure and a review "
     f"citation without a quote, which the stricter axes above close. Table XIV shows the verifier's decisions on "
     f"representative claims for one restaurant; every row is also an automated test.")
table("Verifier decisions on example claims",
      ["Claim (restaurant: own complaint rate 12%, market 20%, risk 9%, rating 3.88★)", "Result"],
      [["\"12 of 100 reviews (12%) complained about service, versus 5% in the six months before; the typical "
        "restaurant in your market sits at 20%.\"", "Pass"],
       ["\"Service complaints cost about 0.12 stars compared with the typical restaurant.\"", "Pass"],
       ["\"20% of your reviews complain about service.\" (market median stated as own rate)", "Statistic"],
       ["\"9% of your reviews complain about service.\" (decline risk stated as a rate)", "Statistic"],
       ["\"Service problems are costing you 3.88 stars.\" (rating stated as a cost)", "Statistic"],
       ["\"Service complaints have tripled: '…'\" (real quote, uncheckable multiplier)", "Statistic"],
       ["Real retrieved review cited with an empty quote and invented content", "Quotation"],
       ["Verbatim praise from a review presented as a complaint", "Relevance"],
       ["Review of this restaurant that the agent never retrieved", "Citation"],
       ["\"Service is clearly the main problem.\" (no checkable figure)", "Substance"]],
      [2.75, 0.65], size=7,
      note="Pass = shown to the owner; otherwise the failing verifier axis. Rejected claims are removed and logged.")
qo = pl["open_model"]
own_calls = qo["mean_tool_calls"] - pl["rule_based"]["mean_tool_calls"]
para(f"The open-model planner shows why the verifier is needed. Qwen 2.5 7B behaved as an agent — beyond the fixed "
     f"analytics calls it made about {own_calls:.0f} further tool calls per plan, mostly to fetch more review evidence "
     f"or compare peers — but only {qo['claims_verified']} of its {qo['claims_total']} final claims "
     f"({qo['groundedness']:.1%}) passed verification. It cited only reviews it had retrieved and quoted them "
     f"verbatim, but {qo['failures_by_axis']['statistic']} claims stated a figure that does not match the quantity "
     f"they name, or an uncheckable multiplier, and {qo['failures_by_axis']['substance']} metric claims stated no "
     f"checkable figure at all. With the earlier, looser numeric check the same planner had scored 69%: the "
     f"difference is exactly the wrong-quantity figures an owner could not spot. A plan took a median of "
     f"{qo['median_latency_s']:.1f} s and about {qo['mean_tokens']:,} tokens on one T4 GPU, at no per-token cost, and "
     f"its actions were largely adapted from the playbook. On this evidence a 7-billion-parameter open model adds "
     f"investigative flexibility but not more specific advice, and is only safe to show owners behind the verifier; "
     f"the Claude planner is implemented and tested but could not be measured without an API key.")

h2("Comparison with Existing Methods and System Performance")
table("Side-by-side comparison with existing methods",
      ["Method", "Predicts", "Use of review text", "Evaluation", "Reported result", "Action plan"],
      [["VD-HMM [4]", "Closure", "One sentiment score per review", "Out-of-sample", "AUC 85.2%", "No"],
       ["CNN–DNN [5]", "Closure", "Aspect-wise topic sentiment", "Random CV split", "≈90% accuracy", "No"],
       ["BERT ABSA [12], [13]", "Aspect sentiment", "Trained on SemEval labels", "SemEval test", "Above ours", "No"],
       ["Radview M2", "Aspect sentiment", "Weak labels from Yelp", "SemEval test", f"Macro-F1 {f3(db['macro_f1'])}", "—"],
       ["Radview M4", "3-month decline", "Aspect complaint signals", "Chronological, purged",
        f"AUROC {lr['auroc']:.3f}", "Verified"]],
      [0.72, 0.52, 0.72, 0.6, 0.5, 0.38], size=6.5,
      note="Results as reported by each work; the tasks differ, so the numbers are not directly comparable (see text).")
para("Table XV places Radview next to the closest published methods. "
     "Direct numerical comparison with [4] and [5] would be misleading: both predict permanent closure, an outcome "
     "with a much stronger and longer-lasting signal than a three-month rating decline, and [5] uses a random rather "
     "than chronological split, which can inflate results through temporal leakage [22]. Qualitatively, Radview adds what "
     "both papers leave as future work — aspect-level causes and recommendations — at the cost of a harder, noisier "
     f"target, and our AUROC of {lr['auroc']:.2f}–{rm4['rating_logreg']['auroc']:.2f} on that target is consistent "
     f"with the gap between the tasks. Operationally, on one T4 GPU fine-tuning DistilBERT took about 5 minutes "
     f"(about an hour on a laptop CPU) and scoring all 152,649 reviews about 14 minutes; training and evaluating all "
     f"seven forecasters takes under a minute; a rule-based plan takes a tenth of a second and an open-model plan "
     f"about {qo['median_latency_s']:.0f} s. The test-gated Docker image is "
     "3.6 GB; the full stack runs on the RONIN t3.large (8 GB) under k3s with about 1 GB of memory in use, and was "
     "first sized on Docker Desktop's single-node cluster (6.8 GiB allocatable); the 43-test suite passes against "
     "the real outputs.")

# =================================================================== V. CONCLUSION
h1("Conclusions")
para("Radview is an end-to-end, containerised system that turns raw restaurant reviews into aspect-level complaint "
     "signals, a calibrated decline forecast with two independent explanations, and a ranked action plan in which "
     "every claim is mechanically verified. On real Yelp data its fine-tuned DistilBERT is the most accurate trained "
     "aspect model, especially for detecting complaints, although a zero-shot open LLM is markedly more accurate at "
     f"about {LLM_SLOWDOWN:.0f} times the cost per sentence; its verifier rejected every fabricated claim, including "
     f"realistic ones built from real figures, and kept a free open model's unsupported claims "
     f"({1 - qo['groundedness']:.0%} of its drafts) away from owners; and complaint spikes mark lasting damage. "
     "Our main contributions are the aspect-to-action pipeline evaluated on real data, the five-axis verifier with an "
     "adversarial test, and a reproducible cloud-native deployment.")
para("The evaluation also contradicted three of our expectations, and we consider reporting this part of the "
     "contribution: a rating-trend logistic regression forecasts decline better than any model that uses aspect "
     "signals; the multi-signal models' early warnings are barely better than random alerts; and complaint spikes do not "
     "anticipate a further decline at monthly granularity. All three hold on a survivor-free selection of restaurants "
     "and, for forecasting, on a pre-pandemic test window. Open limitations are the "
     "COVID-era test window; weak labels that partly borrow the star rating and an evaluation set without a "
     "cleanliness category; low review volumes that make individual rates noisy; attributions and star costs that are "
     "associational rather than causal; a verifier that checks traceability, not truth; and a small open LLM planner "
     f"whose claims pass verification only {qo['groundedness']:.0%} of the time. Future work: train DistilBERT on LLM "
     "labels instead of lexicon rules; hand-label the exported 300-review Yelp sample; extend to several cities; "
     "compare a frontier LLM planner such as Claude with the open model, and collect owner usefulness ratings, which "
     "the dashboard already records.")

# =================================================================== REFERENCES
h1("References", numbered=False)
REFS = [
    "M. Luca, \"Reviews, reputation, and revenue: The case of Yelp.com,\" Harvard Business School, Working Paper 12-016, 2016.",
    "M. Anderson and J. Magruder, \"Learning from the crowd: Regression discontinuity estimates of the effects of an online review database,\" *Econ. J.*, vol. 122, no. 563, pp. 957–989, 2012.",
    "H. G. Parsa, J. T. Self, D. Njite, and T. King, \"Why restaurants fail,\" *Cornell Hotel Restaurant Admin. Quart.*, vol. 46, no. 3, pp. 304–322, 2005.",
    "C. Naumzik, S. Feuerriegel, and M. Weinmann, \"I will survive: Predicting business failures from customer ratings,\" *Marketing Sci.*, vol. 41, no. 1, pp. 188–207, 2022, doi: 10.1287/mksc.2021.1317.",
    "S. S. Thazhackal and V. S. Devi, \"A hybrid deep learning model to predict business closure from reviews and user attributes using sentiment aligned topic model,\" in *Proc. IEEE Symp. Ser. Comput. Intell. (SSCI)*, 2018, pp. 397–404, doi: 10.1109/SSCI.2018.8628823.",
    "M. Hu and B. Liu, \"Mining and summarizing customer reviews,\" in *Proc. 10th ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining*, 2004, pp. 168–177.",
    "M. Pontiki, D. Galanis, J. Pavlopoulos, H. Papageorgiou, I. Androutsopoulos, and S. Manandhar, \"SemEval-2014 Task 4: Aspect based sentiment analysis,\" in *Proc. 8th Int. Workshop Semantic Eval. (SemEval)*, 2014, pp. 27–35.",
    "G. Ganu, N. Elhadad, and A. Marian, \"Beyond the stars: Improving rating predictions using review text content,\" in *Proc. 12th Int. Workshop Web Databases (WebDB)*, Providence, RI, USA, 2009.",
    "J. McAuley and J. Leskovec, \"Hidden factors and hidden topics: Understanding rating dimensions with review text,\" in *Proc. 7th ACM Conf. Recommender Syst. (RecSys)*, 2013, pp. 165–172.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, \"BERT: Pre-training of deep bidirectional transformers for language understanding,\" in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.",
    "V. Sanh, L. Debut, J. Chaumond, and T. Wolf, \"DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter,\" arXiv:1910.01108, 2019.",
    "C. Sun, L. Huang, and X. Qiu, \"Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence,\" in *Proc. NAACL-HLT*, 2019, pp. 380–385.",
    "H. Xu, B. Liu, L. Shu, and P. S. Yu, \"BERT post-training for review reading comprehension and aspect-based sentiment analysis,\" in *Proc. NAACL-HLT*, 2019, pp. 2324–2335.",
    "A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré, \"Snorkel: Rapid training data creation with weak supervision,\" *Proc. VLDB Endowment*, vol. 11, no. 3, pp. 269–282, 2017.",
    "S. W. Roberts, \"Control chart tests based on geometric moving averages,\" *Technometrics*, vol. 1, no. 3, pp. 239–250, 1959.",
    "C. Leys, C. Ley, O. Klein, P. Bernard, and L. Licata, \"Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median,\" *J. Exp. Social Psychol.*, vol. 49, no. 4, pp. 764–766, 2013.",
    "K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, \"Learning phrase representations using RNN encoder–decoder for statistical machine translation,\" in *Proc. EMNLP*, 2014, pp. 1724–1734.",
    "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, \"LightGBM: A highly efficient gradient boosting decision tree,\" in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.",
    "T. Saito and M. Rehmsmeier, \"The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,\" *PLoS ONE*, vol. 10, no. 3, e0118432, 2015.",
    "G. W. Brier, \"Verification of forecasts expressed in terms of probability,\" *Monthly Weather Rev.*, vol. 78, no. 1, pp. 1–3, 1950.",
    "J. Platt, \"Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,\" in *Advances in Large Margin Classifiers*, A. Smola et al., Eds. Cambridge, MA, USA: MIT Press, 1999, pp. 61–74.",
    "S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, \"Leakage in data mining: Formulation, detection, and avoidance,\" *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, Art. 15, 2012.",
    "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 4765–4774.",
    "G. Erion, J. D. Janizek, P. Sturmfels, S. M. Lundberg, and S.-I. Lee, \"Improving performance of deep learning models with axiomatic attribution priors and expected gradients,\" *Nature Mach. Intell.*, vol. 3, no. 7, pp. 620–631, 2021.",
    "S. Wachter, B. Mittelstadt, and C. Russell, \"Counterfactual explanations without opening the black box: Automated decisions and the GDPR,\" *Harvard J. Law Technol.*, vol. 31, no. 2, pp. 841–887, 2018.",
    "S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, \"ReAct: Synergizing reasoning and acting in language models,\" in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023.",
    "P. Lewis et al., \"Retrieval-augmented generation for knowledge-intensive NLP tasks,\" in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 9459–9474.",
    "Z. Ji et al., \"Survey of hallucination in natural language generation,\" *ACM Comput. Surv.*, vol. 55, no. 12, Art. 248, 2023.",
    "T. Gao, H. Yen, J. Yu, and D. Chen, \"Enabling large language models to generate text with citations,\" in *Proc. EMNLP*, 2023, pp. 6465–6488.",
    "B. Burns, B. Grant, D. Oppenheimer, E. Brewer, and J. Wilkes, \"Borg, Omega, and Kubernetes,\" *Commun. ACM*, vol. 59, no. 5, pp. 50–57, 2016.",
]
for i, ref in enumerate(REFS, start=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(1)
    add_runs(p, f"[{i}]  {ref}", size=8)

# =================================================================== CONTRIBUTIONS & REFLECTIONS
back = doc.add_section(WD_SECTION.NEW_PAGE)
set_columns(back, 1)
h1("Individual Workloads and Contributions", numbered=False)
para("[[To be completed and agreed by the group. The module split below follows the proposal's six modules as a "
     "suggestion only — replace it with what each member actually did, including report sections and the part of "
     "the presentation each member delivers.]]", indent=False)
MEMBERS = ["Yeo Zhang Han", "Daniel Leong", "Peh Xue Ying", "Yang Jiajun Joven", "Wong Wei Xuen", "Vijayanand Jeeva"]
table("Individual contributions",
      ["Member", "Main technical contribution", "Report sections", "Presentation part"],
      [[m, "[[e.g. M1–M6 module / dashboard / deployment]]", "[[e.g. Sec. III-B]]", "[[e.g. demo of …]]"] for m in MEMBERS],
      [1.35, 2.9, 1.25, 1.4], size=9)
para("**Use of AI tools.** [[Declare as required by the module's academic-integrity policy, e.g.: \"Parts of the "
     "implementation, documentation and figures were developed with the assistance of an AI coding assistant "
     "(Claude Code). All design decisions, results and text were reviewed, verified and are owned by the group.\" — "
     "adjust to reflect actual use.]]", indent=False, space_after=6)
h1("Reflections", numbered=False)
for m in MEMBERS:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    add_runs(p, f"**{m}.** ")
    add_runs(p, "[[Personal reflection on the learning journey (≈100–150 words): what you worked on, what you learned "
                "about cloud-based data science, AI and teamwork, what was hardest, and what you would do differently.]]")

doc.save(OUT)
print("wrote", OUT)
