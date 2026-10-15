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
m5o = json.loads((Y / "m5_open_model_summary.json").read_text(encoding="utf8"))
m4s = json.loads((Y / "m4_evaluation_synthetic.json").read_text())
m4pre = json.loads((Y / "m4_evaluation_test_end_2019-12.json").read_text())

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


_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
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
    "Online ratings drive restaurant revenue, yet owners usually discover a reputation problem only after their average "
    "rating has already fallen, and the aggregate score never says what to fix. Existing research predicts permanent "
    "business failure from rating sequences, and commercial review analytics report overall sentiment; both describe "
    "the past and stop at diagnosis. We present Radview, a cloud-native system that warns owners of an impending rating "
    "decline and tells them, with cited review evidence, which operational problem to address first. Radview streams "
    "the Yelp Open Dataset, labels every review's sentiment toward food, service, cleanliness, price and ambience with "
    "a DistilBERT model fine-tuned by weak supervision, tracks aspect-level complaint rates with robust control charts, "
    "and forecasts a statistically significant 0.3-star decline over the next three months with a calibrated "
    "PyTorch GRU over twelve-month feature sequences. Forecasts are explained twice, by SHAP and by a counterfactual "
    "intervention, and an agent built with LangGraph turns them into a ranked action plan whose every claim is checked "
    "by a mechanical five-axis verifier before an owner sees it. The system runs as Docker containers on Kubernetes in AWS "
    "behind a REST API and an owner dashboard. On 152,649 reviews of 397 Philadelphia restaurants (2016–2021), the "
    "aspect model reaches macro-F1 0.737 and complaint-F1 0.567 on human-labelled SemEval-2014 sentences, ahead of "
    "TF-IDF and lexicon baselines; all 265 claims in 53 real action plans pass verification while 100% of claims from "
    "an adversarial planner are rejected, and a free open model (Qwen 2.5 7B) acting as the agent gets only 69% of "
    "its claims verified. On an out-of-time, COVID-era test period the GRU (AUROC 0.727) is the best "
    "learned multi-signal model, but a rating-trend logistic regression is marginally better (0.738), and complaint "
    "spikes do not anticipate declines at monthly granularity. We report these negative results and discuss why."),
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
    "(the star rating only fills in polarity for opinion-less sentences in the training labels) feeds complaint monitoring, a calibrated decline forecaster with two independent "
    "explanations, and an action planner — evaluated on 152,649 real Yelp reviews and on human-labelled "
    "SemEval-2014 sentences.",
    "**A grounded agent with a five-axis verifier.** A LangGraph agent in which an LLM — Claude, or a free open-weight "
    "model served on our own GPU — chooses which analytics tools to call, with a deterministic planner as fallback "
    "and baseline; every claim is checked for citation, verbatim "
    "quotation, aspect relevance, numeric fidelity and substance, and an adversarial planner demonstrates that "
    "fabrications are rejected.",
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
     "model that generalises beyond the heuristics; unlike [12], [13], our model never sees SemEval training labels.")
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
       ["Aspect polarity from text", "No", "Yes", "Varies", "Yes (DistilBERT)"],
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
     "from the business file in seconds, so a user can choose a slice before ingesting. Table II gives the resulting "
     "dataset; a Yelp-format simulator with injected deterioration episodes provides a second, controlled dataset.")
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
     "(*not clean* is a complaint). Only when a sentence carries no opinion word do we fall back on the review's star "
     "rating. Generic words such as *ordered* or *dinner* are excluded from the seed lists after we found they "
     "copied the star rating onto neutral sentences. On 40,000 weakly labelled reviews, food is mentioned in 85% of "
     "reviews and cleanliness in only 7%, so the label distribution is highly imbalanced.")
para("Three models are trained on these labels. The *lexicon rule* is the labeller itself. The *TF-IDF baseline* "
     "fits one class-balanced logistic regression per aspect on word uni- and bigrams. The proposed model fine-tunes "
     "DistilBERT [11]: the [CLS] representation feeds a linear layer with 5 × 3 outputs, one softmax head per aspect, "
     "trained with class-weighted cross-entropy (weights ∝ inverse class frequency^0.5) for two epochs on 6,000 "
     "reviews (sequence length 128, AdamW, learning rate 3×10⁻⁵, warm-up and linear decay). For serving on CPU the "
     "encoder's linear layers are dynamically quantised to int8, which cut inference time by about 30% in our "
     "measurements. The model with the best macro-F1 on the benchmark is selected automatically and scores every review.")

h2("M3 — Emerging Complaint Detection")
para("For restaurant *i*, aspect *a* and month *t*, the complaint rate is the share of reviews labelled negative on "
     "*a*. Monthly counts are sparse, so we smooth the numerator and denominator separately with an EWMA (span 3) "
     "and take their ratio, then compute a robust z-score against the restaurant's *own* previous twelve months:")
para("*z*_{t} = (*r*_{t} − median(*r*_{t−12}, …, *r*_{t−1})) / max(1.4826·MAD, binomial s.e., 0.04).",
     indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
para("The floor stops a perfectly stable history from producing an infinite score, and the binomial term widens the "
     "scale for low-volume restaurants. An aspect is flagged when *z* > 2.5 and at least two negative mentions occurred "
     "in the last two months, so that \"100% of two reviews\" cannot fire an alert. For comparison, the same detector "
     "is applied to the smoothed star rating itself (a *rating alarm*).")

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
     "are defined as *lift* over the training base rate (≥1.5× medium, ≥2× high) because calibrated probabilities "
     "for a 6% event rarely exceed 15%. Forecasts are explained in two ways: SHAP values from the expected-gradients "
     "estimator [23], [24], summed over time and grouped by aspect, and a counterfactual that resets one aspect's "
     "features to the market median for the whole window and re-runs the network [25]. A ridge regression of monthly "
     "mean stars on aspect complaint rates converts \"how far above the market\" into an estimated star cost.")

h2("M5 — Grounded Agentic Prioritisation")
para("M5 is a LangGraph state machine (Fig. 2). It first calls four tools — *query_metrics*, *run_forecast*, "
     "*get_evidence_reviews* and *compare_peers* — each a thin, read-only function over the M2–M4 outputs scoped to "
     "one restaurant. Aspects are ranked by priority = (0.7·impact + 0.3·urgency) / effort, where impact averages each "
     "aspect's share of positive SHAP and counterfactual attribution, urgency combines the M3 z-score and complaint "
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
     "exists, belongs to this restaurant and was retrieved in this run), *quotation* (the quote appears verbatim), "
     "*relevance* (M2 labels the review negative on the aspect and the quoted sentence alone is a complaint about it), "
     "*statistic* (every percentage is within two points, and every star figure within 0.05, of a tool output) and "
     "*substance* (a metric claim must state a checkable figure). Failing claims are removed, and the share that "
     "passes is the plan's groundedness. A hallucinating planner that fabricates one claim per failure mode is part "
     "of the test suite.")

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

# =================================================================== IV. EXPERIMENTS
h1("Experiment Evaluations and Performance Comparisons")
h2("Experimental Setup")
sp = m4["split"]
para(f"All experiments ran on a 16-thread laptop CPU without a GPU. Forecasting uses a strict chronological split: "
     f"training months {sp['train'].replace('..', '–')}, validation {sp['valid'].replace('..', '–')} and test "
     f"{sp['test'].replace('..', '–')} ({sp['n_train']:,}, {sp['n_valid']:,} and {sp['n_test']:,} labelled "
     f"restaurant-months; {sp['pos_rate_train']:.1%} and {sp['pos_rate_test']:.1%} positive in training and test). "
     f"The test window coincides with the COVID-19 pandemic, a regime the models never saw in training. Metrics follow "
     f"Section II: macro-F1 and complaint-F1 for M2; precision, lift and lead time for M3; AUROC, PR-AUC, Brier score, "
     f"precision and recall in the riskiest 10% of restaurant-months, and early warning at a fixed 15% alert budget "
     f"for M4; and groundedness for M5. Fixed random seeds make all reported numbers reproducible.")

h2("Aspect Classification (M2)")
res = {r["model"]: r for r in m2["results"]["semeval2014"] if "macro_f1" in r}
lab = {"lexicon_rule": "Lexicon rule (weak labeller)", "tfidf_logreg": "TF-IDF + logistic regression",
       "distilbert_finetuned": "**DistilBERT, fine-tuned (ours)**"}
table("Aspect sentiment on SemEval-2014 restaurant test set",
      ["Model", "Macro-F1", "Compl.-F1", "ms/sent."],
      [[lab[k], f3(res[k]["macro_f1"]), f3(res[k]["complaint_f1"]), f"{res[k]['inference_ms_per_review']:.2f}"]
       for k in ("lexicon_rule", "tfidf_logreg", "distilbert_finetuned")],
      [1.75, 0.55, 0.6, 0.5],
      note="800 human-labelled sentences; food, service, price, ambience (SemEval has no cleanliness category). "
           "No model saw SemEval training data.")
para("Table IV and Fig. 4 compare the three aspect models on 800 human-labelled SemEval-2014 sentences that none of "
     "them was trained on. Both learned models improve substantially on the lexicon rule that produced their training "
     "labels (macro-F1 0.737 and 0.727 versus 0.660), which confirms that weak supervision generalises beyond its own "
     "heuristics. DistilBERT is the best model overall, and its advantage is concentrated where Radview needs it: "
     "complaint-F1 rises from 0.512 to 0.567 (+11% relative), with the largest gains on service (0.689 → 0.746) and "
     "ambience (0.227 → 0.348). TF-IDF remains slightly better on price (0.773 vs 0.748 macro-F1), where complaints use "
     "a small, distinctive vocabulary (*overpriced*, *expensive*). Ambience is hardest for every model: it is mentioned "
     "in only 28% of reviews and its complaints are lexically diverse. Fully supervised BERT models trained on "
     "SemEval's own training set report higher scores [12], [13]; ours is a transfer result from noisy Yelp labels, "
     "and DistilBERT costs about 6 ms per sentence on CPU against 0.06 ms for TF-IDF.")
figure(FIGS / "fig3_m2.png",
       "Per-aspect F1 on SemEval-2014 for the lexicon rule, TF-IDF and fine-tuned DistilBERT. Left: three-class "
       "macro-F1. Right: F1 for detecting complaints (negative class).")

h2("Emerging Complaint Detection (M3)")
cs, ra = m3["complaint_spike"], m3["rating_alarm"]
table("Do alerts anticipate a decline in the next three months?",
      ["Detector", "Flagged", "Precision", "Lift", "Onsets warned", "Lead (mo)"],
      [[n, f"{d['flag_rate']:.1%}", f"{d['precision']:.1%}", f"{d['lift']:.2f}", f"{d['onsets_detected_pct']}%",
        f"{d['median_lead_months']:g}"] for n, d in (("Complaint spike", cs), ("Rating alarm", ra))],
      [0.92, 0.44, 0.52, 0.34, 0.62, 0.52], size=7,
      note=f"Base rate {m3['base_rate']:.1%}; {cs['onsets']} decline onsets; lift = precision / base rate.")
para(f"Table V tests the proposal's central hypothesis: that aspect complaint spikes are a *leading* indicator. On "
     f"real data they are not. Restaurant-months with a complaint spike are followed by a decline only "
     f"{cs['precision']:.1%} of the time, below the {m3['base_rate']:.1%} base rate (lift {cs['lift']:.2f}), and only "
     f"{cs['onsets_detected_pct']}% of decline onsets are preceded by a spike in the previous six months. The rating "
     f"alarm is worse still (lift {ra['lift']:.2f}). We believe the main reason is regression to the mean: a spike "
     f"typically arrives when the rating has *already* dropped, and because the label compares the next three months "
     f"with the past six, the months after a spike tend to recover rather than fall further. M3 is therefore more "
     f"useful as a *diagnostic* — it tells the owner which aspect broke — than as a forecaster, which is how the agent uses it.")

h2("Rating-Decline Forecasting (M4)")
names = {"majority": "Majority (base rate)", "current_negativity": "Current negativity (no model)",
         "rating_logreg": "Rating trend, LogReg", "rating_lgbm": "Rating trend, LightGBM",
         "sentiment_lgbm": "+ overall sentiment, LightGBM", "radview_lgbm": "All features, LightGBM",
         "radview_gru": "**All features, GRU (ours)**"}
rows = []
for k, r in m4["results"].items():
    ew = r.get("early_warning") or {}
    rows.append([names[k], f3(r["auroc"]), f3(r["pr_auc"]), f"{r['brier']:.4f}", f3(r["precision_at_10pct"]),
                 f"{ew['detected_pct']}%" if ew.get("detected_pct") is not None else "–"])
table("Decline forecasting on the out-of-time test period",
      ["Model", "AUROC", "PR-AUC", "Brier", "P@10%", "Warned"], rows, [1.22, 0.42, 0.44, 0.46, 0.42, 0.44], size=7,
      note="All models Platt-calibrated on validation. P@10% = precision in the riskiest 10% of restaurant-months; "
           "Warned = decline onsets flagged within 6 months at a 15% alert budget.")
g, lr = m4["results"]["radview_gru"], m4["results"]["rating_logreg"]
para(f"Table VI and Fig. 5 compare seven forecasters, from a constant base rate to the full GRU. Every learned model "
     f"beats the trivial baselines, and the GRU is the strongest model that uses all signals: AUROC {g['auroc']:.3f} "
     f"and PR-AUC {g['pr_auc']:.3f}, ahead of every LightGBM variant (at most "
     f"{m4['results']['sentiment_lgbm']['auroc']:.3f} and {m4['results']['sentiment_lgbm']['pr_auc']:.3f}), with "
     f"precision {g['precision_at_10pct']:.1%} in its riskiest decile — about three times the {sp['pos_rate_test']:.1%} "
     f"base rate. However, the simplest learned model, a logistic regression on rating-trend features, is marginally "
     f"better on every ranking metric (AUROC {lr['auroc']:.3f}, PR-AUC {lr['pr_auc']:.3f}, and the most decline "
     f"onsets warned). Fig. 5 shows why: its precision is higher at low recall, where an owner would set an alert "
     f"threshold. Adding aspect features to LightGBM does not help either (AUROC "
     f"{m4['results']['radview_lgbm']['auroc']:.3f} vs {m4['results']['rating_lgbm']['auroc']:.3f} without them). "
     f"Consistent with M3, on this data *what* customers complain about does not improve a three-month decline forecast "
     f"beyond the rating trend itself; the most important LightGBM features by mean |SHAP| are the 6-month and "
     f"cumulative ratings. The COVID-era test window — review volume fell by more than half in 2020 — penalises the "
     f"higher-capacity models most; Section IV-E therefore repeats the comparison on a pre-pandemic test window.")
figure(FIGS / "fig4_m4_pr.png",
       "Precision–recall curves on the test period. The dotted line is the 7.0% base rate. The rating-trend logistic "
       "regression dominates at low recall; the GRU is the strongest of the multi-signal models.")
syn_g = m4s["injected_episodes"]["radview_gru"]
agr = m4["attribution_agreement"]
para(f"On the simulator, where the timing of injected deterioration episodes is known, the GRU caught "
     f"{syn_g['detected_pct']}% of {syn_g['episodes']} episodes and {syn_g['before_visible_pct']}% before the drop "
     f"was visible in the three-month rating — close to the rating-trend logistic regression (23.9%) and ahead of "
     f"the LightGBM models (13.4–17.9%). The two explanation methods agree on the top aspect for "
     f"{agr['all']['top_aspect_agreement']:.0%} of the {agr['all']['n']} restaurants (mean Spearman correlation "
     f"{agr['all']['mean_spearman']:.2f}), which suggests the aspect ranking shown to owners is not an artefact of one "
     f"method. The star-impact model assigns the largest costs to food ({agr['star_impact_coefficients']['food']:.2f} "
     f"stars per unit complaint rate), service ({agr['star_impact_coefficients']['service']:.2f}) and cleanliness "
     f"({agr['star_impact_coefficients']['cleanliness']:.2f}); ambience receives a small positive coefficient, a "
     f"correlation artefact, so star costs are clipped at zero.")

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
     f"(Table VII). Absolute scores fall for every model — there is half as much training data — but the ordering is "
     f"unchanged: the rating-trend logistic regression remains best (AUROC {pre['rating_logreg']['auroc']:.3f}, "
     f"PR-AUC {pre['rating_logreg']['pr_auc']:.3f}), the GRU second ({pre['radview_gru']['auroc']:.3f}, "
     f"{pre['radview_gru']['pr_auc']:.3f}), and the LightGBM models degrade most (AUROC "
     f"{min(pre[k]['auroc'] for k in ('rating_lgbm', 'sentiment_lgbm', 'radview_lgbm')):.3f}–"
     f"{max(pre[k]['auroc'] for k in ('rating_lgbm', 'sentiment_lgbm', 'radview_lgbm')):.3f}). The advantage of the "
     f"simple model is therefore not a COVID artefact. It is better explained by sample size: with about 6% positives, "
     f"a few hundred decline events per year are too few for high-capacity models to learn interactions that a "
     f"linear model on well-chosen trend features does not already capture. The sequence model's inductive bias — "
     f"reading the trajectory rather than hand-made windows — narrows but does not close that gap.")

h2("Discussion and Threats to Validity")
para("*Construct validity.* Our decline label is a statistical construct; a restaurant owner may care about a smaller "
     "but persistent drop, or about review volume falling. The significance test makes positives more meaningful but "
     "also rarer, which limits what any model can learn. *Internal validity.* All features are computed from past data "
     "and the split is chronological (both enforced by tests), but the aspect labels come from a model trained on "
     "reviews spanning the whole period; because M2 is trained on weak labels rather than on outcomes, we judge the "
     "resulting leakage negligible, and a stricter design would retrain M2 on training-period reviews only. "
     "*External validity.* The data cover one city and its 400 busiest restaurants, which are larger and better "
     "reviewed than the typical independent restaurant Radview targets; a quieter restaurant produces exactly the "
     "sparse, noisy signals that our small-sample caveat warns about. *Measurement.* SemEval sentences are shorter and "
     "cleaner than Yelp reviews and have no cleanliness category, so M2's in-domain accuracy is unknown until the "
     "exported Yelp sample is hand-labelled. Finally, the verifier proves traceability, not truth: a plan that "
     "faithfully cites a miscalibrated forecast still passes.")

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
para(f"Across all 53 at-risk restaurants, the rule-based planner produced 265 claims and all of them passed the "
     f"verifier, with a median latency of 0.27 s per plan. This is expected by construction — the planner only "
     f"restates tool outputs — so the verifier's value is shown by the adversarial planner (Table VIII): every one of its "
     f"{pl['adversarial']['claims_total']} fabricated claims was rejected, and every axis fired, from invented review "
     f"IDs (citation) to quoted praise passed off as complaints (relevance). Inspecting real plans exposed two issues "
     f"we then fixed: rates such as \"33% of reviews\" can mean one of three reviews in the sparse late-2021 data, so "
     f"plans now state counts and add a small-sample caveat below ten reviews; and generic keywords led some actions to "
     f"default to generic advice.")
qo = pl["open_model"]
ex = m5o["rejected_examples_model_written"]
para(f"The open-model planner shows why the verifier is needed. Qwen 2.5 7B behaved as an agent — beyond the fixed "
     f"analytics calls it chose {m5o['model_chosen_tool_calls_per_plan']} further tool calls per plan, mostly to fetch "
     f"more review evidence or compare peers — but only {qo['claims_verified']} of its {qo['claims_total']} final "
     f"claims ({qo['groundedness']:.1%}) passed verification. Of the {m5o['restaurants']} plans, "
     f"{m5o['plans_fully_verified']} were fully grounded and {m5o['plans_redrafted_after_verifier_feedback']} needed a "
     f"redraft after verifier feedback, but {m5o['plans_with_no_verified_claim']} ended with no verified claim. The "
     f"rejected claims were mainly paraphrases presented as quotations ({qo['failures_by_axis']['quotation']}), metric "
     f"claims with no checkable figure ({qo['failures_by_axis']['substance']}; e.g. \"{ex['substance']['claim']}\") and "
     f"citations of reviews that were never retrieved, including invented identifiers such as \"r123\"; one stated a "
     f"statistic the tools never returned (\"{ex['statistic']['claim']}\"). Owners therefore saw "
     f"{m5o['owner_visible_items_with_verified_quote']} action items backed by a verified quotation, against "
     f"{m5o['rule_based_items_with_verified_quote']} from the rule-based planner. A plan took a median of "
     f"{qo['median_latency_s']:.1f} s and about {qo['mean_tokens']:,} tokens on one T4 GPU, at no per-token cost, and "
     f"its actions were largely adapted from the playbook. On this evidence a 7-billion-parameter open model adds "
     f"investigative flexibility but not more specific advice, and is only safe to show owners behind the verifier; "
     f"the Claude planner is implemented and tested but could not be measured without an API key.")

h2("Comparison with Existing Methods and System Performance")
para("Direct numerical comparison with [4] and [5] would be misleading: both predict permanent closure, an outcome "
     "with a much stronger and longer-lasting signal than a three-month rating decline, and [5] uses a random rather "
     "than chronological split, which can inflate results through temporal leakage [22]. Qualitatively, Radview adds what "
     "both papers leave as future work — aspect-level causes and recommendations — at the cost of a harder, noisier "
     "target, and our AUROC of 0.73–0.74 on that target is consistent with the gap between the tasks. Operationally, "
     "fine-tuning DistilBERT took 54 minutes and scoring all 152,649 reviews 52 minutes on CPU; training and evaluating "
     "all seven forecasters takes under two minutes; a rule-based plan takes a quarter of a second and an "
     "open-model plan about 15 s on a T4 GPU. The test-gated Docker image is "
     "3.6 GB; the full stack runs on the RONIN t3.large (8 GB) under k3s with about 1 GB of memory in use, and was "
     "first sized on Docker Desktop's single-node cluster (6.8 GiB allocatable); the 27-test suite passes against "
     "the real outputs.")

# =================================================================== V. CONCLUSION
h1("Conclusions")
para("Radview is an end-to-end, containerised system that turns raw restaurant reviews into aspect-level complaint "
     "signals, a calibrated decline forecast with two independent explanations, and a ranked action plan in which "
     "every claim is mechanically verified. On real Yelp data its fine-tuned DistilBERT is the most accurate aspect "
     "model on human-labelled SemEval sentences, especially for detecting complaints; its agent produced 53 fully "
     "grounded plans while rejecting every fabricated claim, and kept a free open model's unsupported claims "
     "(31% of its drafts) away from owners; and its GRU is the best multi-signal forecaster. Our "
     "main contributions are the aspect-to-action pipeline evaluated on real data, the five-axis verifier with an "
     "adversarial test, and a reproducible cloud-native deployment.")
para("The evaluation also contradicted two of our expectations, and we consider reporting this part of the "
     "contribution: a rating-trend logistic regression forecasts decline marginally better than any model that uses "
     "aspect signals, and complaint spikes do not anticipate declines at monthly granularity. Open limitations are the "
     "COVID-era test window; weak labels and an evaluation set without a cleanliness category; low review volumes that "
     "make individual rates noisy; attributions and star costs that are associational rather than causal; a verifier "
     "that checks traceability, not truth; and a small open LLM planner whose claims pass verification only "
     "69% of the time. Future work: redefine the spike target as lasting damage relative to the pre-spike baseline; hand-label "
     "the exported 300-review Yelp sample; extend to several cities; compare a frontier LLM planner such as Claude with the open model, and collect owner "
     "usefulness ratings, which the dashboard already records.")

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
