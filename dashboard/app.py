"""M6 - Radview owner dashboard (Streamlit).

Pages
    Portfolio          every monitored restaurant ranked by decline risk
    Restaurant         one restaurant: action plan with cited evidence, risk drivers,
                       rating & complaint trends, peer comparison, review explorer
    Model performance  M2-M5 evaluation results and model comparisons
    About              architecture and how the pipeline works

Run:  streamlit run dashboard/app.py
      (set RADVIEW_API_URL=http://localhost:8000 to read from the API instead of in-process)
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `streamlit run dashboard/app.py`
from dashboard.client import Client  # noqa: E402

ASPECTS = ["food", "service", "cleanliness", "price", "ambience"]
# Categorical slots in fixed order (validated palette); aspects always keep the same colour.
ASPECT_COLORS = {"food": "#2a78d6", "service": "#eb6834", "cleanliness": "#1baf7a", "price": "#eda100",
                 "ambience": "#e87ba4"}
# Status colours are reserved for risk state and always paired with an icon + label.
STATUS = {"high": ("#d03b3b", "🔴 High"), "medium": ("#ec835a", "🟠 Medium"), "low": ("#0ca30c", "🟢 Low")}
INCREASES, DECREASES = "#e34948", "#2a78d6"   # diverging poles for SHAP direction
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
MODEL_LABELS = {"rating_logreg": "Rating trend · LogReg", "rating_lgbm": "Rating trend · LightGBM",
                "sentiment_lgbm": "+ Overall sentiment · LightGBM", "radview_lgbm": "Radview features · LightGBM",
                "radview_gru": "Radview · PyTorch GRU"}
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

st.set_page_config(page_title="Radview", page_icon="📡", layout="wide")


def _theme():
    """Recessive axes & grid; text in ink colours, never series colours."""
    return {"config": {"view": {"stroke": None}, "font": "sans-serif",
                       "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": GRID, "domainColor": GRID,
                                "tickColor": GRID, "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": "normal"},
                       "legend": {"labelColor": INK, "titleColor": MUTED, "orient": "top", "labelFontSize": 11},
                       "header": {"labelColor": INK, "titleColor": MUTED}}}


alt.themes.register("radview", _theme)
alt.themes.enable("radview")


@st.cache_resource
def client() -> Client:
    return Client()


@st.cache_data(ttl=600, show_spinner=False)
def load(kind: str, *args):
    return getattr(client(), kind)(*args)


def pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def status_badge(level: str) -> str:
    color, label = STATUS[level]
    return f"<span style='background:{color}1f;color:{INK};border:1px solid {color};border-radius:12px;" \
           f"padding:2px 10px;font-size:0.85rem;white-space:nowrap'>{label} risk</span>"


# ------------------------------------------------------------------ pages
def page_portfolio():
    meta = load("meta")
    rows = pd.DataFrame(load("restaurants"))
    st.title("📡 Radview — reputation early warning")
    st.caption(f"Data: **{meta['source']}** · latest complete month **{meta['latest_month']}** · "
               f"{meta['restaurants']} restaurants · {meta['reviews']:,} reviews analysed")

    c1, c2, c3 = st.columns([2, 2, 3])
    cuisines = sorted(rows["cuisine"].dropna().unique())
    pick_cuisine = c1.multiselect("Cuisine", cuisines, placeholder="All cuisines")
    pick_level = c2.multiselect("Risk level", ["high", "medium", "low"], placeholder="All levels",
                                format_func=lambda l: STATUS[l][1])
    query = c3.text_input("Search restaurant", placeholder="name contains…")
    view = rows.copy()
    if pick_cuisine:
        view = view[view["cuisine"].isin(pick_cuisine)]
    if pick_level:
        view = view[view["risk_level"].isin(pick_level)]
    if query:
        view = view[view["name"].str.contains(query, case=False, na=False)]

    k = st.columns(5)
    k[0].metric("Restaurants monitored", len(view))
    k[1].metric("🔴 High risk", int((view["risk_level"] == "high").sum()))
    k[2].metric("🟠 Medium risk", int((view["risk_level"] == "medium").sum()))
    k[3].metric("With complaint spikes", int(view["spiking_aspects"].map(len).gt(0).sum()))
    k[4].metric("Avg rating (3 mo)", f"{view['rating_3m'].mean():.2f} ★" if len(view) else "–")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Restaurants ranked by decline risk")
        st.caption("Risk = calibrated probability that the average rating drops ≥0.3★ over the next 3 months. "
                   "Click a row to open the restaurant.")
        table = view.assign(
            level=view["risk_level"].map(lambda l: STATUS[l][1]),
            spikes=view["spiking_aspects"].map(lambda s: ", ".join(s) if len(s) else "—"),
        )[["name", "cuisine", "level", "risk", "rating_3m", "rating_6m", "reviews_3m", "top_aspect", "spikes", "business_id"]]
        event = st.dataframe(
            table, hide_index=True, use_container_width=True, height=460, on_select="rerun", selection_mode="single-row",
            column_config={
                "name": "Restaurant", "cuisine": "Cuisine", "level": "Level",
                "risk": st.column_config.ProgressColumn("Decline risk", min_value=0, max_value=1, format="percent"),
                "rating_3m": st.column_config.NumberColumn("Rating 3 mo", format="%.2f ★"),
                "rating_6m": st.column_config.NumberColumn("Rating 6 mo", format="%.2f ★"),
                "reviews_3m": st.column_config.NumberColumn("Reviews 3 mo"),
                "top_aspect": "Main driver", "spikes": "Complaint spikes", "business_id": None})
        if event.selection.rows:
            st.session_state["bid"] = table.iloc[event.selection.rows[0]]["business_id"]
            st.switch_page(PAGES["restaurant"])
    with right:
        st.subheader("Risk distribution")
        hist = alt.Chart(view).mark_bar(color="#2a78d6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                        stroke="white", strokeWidth=2).encode(
            x=alt.X("risk:Q", bin=alt.Bin(step=0.05), title="Decline risk", axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", title="Restaurants"),
            tooltip=[alt.Tooltip("risk:Q", bin=alt.Bin(step=0.05), title="Risk band", format=".0%"),
                     alt.Tooltip("count():Q", title="Restaurants")]).properties(height=200)
        st.altair_chart(hist, use_container_width=True)
        st.subheader("Main risk driver")
        drivers = view["top_aspect"].value_counts().rename_axis("aspect").reset_index(name="n")
        bars = alt.Chart(drivers).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("aspect:N", sort="-x", title=None),
            x=alt.X("n:Q", title="Restaurants"),
            color=alt.Color("aspect:N", scale=alt.Scale(domain=ASPECTS, range=list(ASPECT_COLORS.values())), legend=None),
            tooltip=[alt.Tooltip("aspect:N", title="Aspect"), alt.Tooltip("n:Q", title="Restaurants")]).properties(height=170)
        st.altair_chart(bars + bars.mark_text(align="left", dx=4, color=MUTED).encode(text="n:Q", color=alt.value(MUTED)),
                        use_container_width=True)


def _plan_section(bid: str):
    with st.spinner("Running the agent…"):
        plan = load("plan", bid)
    v = plan.get("verification", {})
    top = st.columns([4, 1])
    top[0].markdown(f"**{plan['summary']}**")
    if top[1].button("↻ Regenerate", help="Re-run the agent (uses Claude if an API key is configured)"):
        load.clear()
        client().plan(bid, refresh=True)
        st.rerun()
    st.caption(f"Drafted by: {'Claude (LLM)' if plan.get('mode') == 'llm' else 'deterministic planner'} · "
               f"✅ {v.get('claims_verified', 0)}/{v.get('claims_total', 0)} claims verified against review evidence "
               f"· generated {plan.get('generated_at', '')}")
    if not plan["items"]:
        st.info("No pressing issues found — nothing to fix right now.")
    for it in plan["items"]:
        with st.container(border=True):
            color = ASPECT_COLORS.get(it["aspect"], MUTED)
            st.markdown(f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};"
                        f"margin-right:6px'></span>**#{it['rank']} · {it['aspect'].capitalize()}** — {it['headline']}",
                        unsafe_allow_html=True)
            st.caption(it["why"])
            ev, act = st.columns([3, 2])
            with ev:
                st.markdown("**Evidence**")
                for c in it["claims"]:
                    if c.get("type") == "review":
                        st.markdown(f"> {c['text']}  \n> ✅ verified · `{', '.join(c.get('review_ids', []))}`")
                    else:
                        st.markdown(f"📊 {c['text']} ✅")
            with act:
                st.markdown("**Recommended actions**")
                for a in it["actions"]:
                    st.markdown(f"- {a}")
                if it.get("sub_themes"):
                    st.caption("Themes in complaints: " + ", ".join(it["sub_themes"]))
    with st.expander("Rate this action plan (helps us measure usefulness)"):
        with st.form(f"fb-{bid}", clear_on_submit=True):
            score = st.slider("How useful is this plan?", 1, 5, 4)
            comment = st.text_input("Comment (optional)")
            if st.form_submit_button("Submit rating"):
                summary = client().feedback(bid, score, comment)
                st.success(f"Thanks! Average usefulness so far: {summary['mean_usefulness']} / 5 ({summary['n']} ratings)")


def page_restaurant():
    rows = pd.DataFrame(load("restaurants"))
    ids = rows["business_id"].tolist()
    names = dict(zip(rows["business_id"], rows["name"] + " · " + rows["risk_level"].map(lambda l: STATUS[l][1])))
    default = st.session_state.get("bid", ids[0])
    bid = st.selectbox("Restaurant", ids, index=ids.index(default) if default in ids else 0, format_func=names.get)
    st.session_state["bid"] = bid
    d = load("restaurant", bid)
    info, m, fc, peers = d["info"], d["metrics"], d["forecast"], d["peers"]

    st.markdown(f"## {info['name']} &nbsp; {status_badge(fc['risk_level'])}", unsafe_allow_html=True)
    st.caption(f"{info['cuisine']} · {info['address']}, {info['city']} · month {m['month']}")
    k = st.columns(4)
    k[0].metric("Decline risk (next 3 mo)", pct(fc["risk"]))
    delta = None if m["rating_3m"] is None or m["rating_6m"] is None else round(m["rating_3m"] - m["rating_6m"], 2)
    k[1].metric("Rating, last 3 months", f"{m['rating_3m']:.2f} ★" if m["rating_3m"] else "–",
                delta=f"{delta:+.2f} vs 6 mo" if delta is not None else None)
    k[2].metric("All-time rating", f"{m['rating_all_time']:.2f} ★" if m["rating_all_time"] else "–")
    k[3].metric("Reviews, last 3 months", m["reviews_3m"])

    tab_plan, tab_trend, tab_peer, tab_rev = st.tabs(["🧭 Action plan", "📈 Trends", "👥 Peers", "💬 Reviews"])
    with tab_plan:
        left, right = st.columns([3, 2])
        with left:
            _plan_section(bid)
        with right:
            st.markdown("**What drives the risk score** (SHAP, logit units)")
            attr = pd.DataFrame([{"driver": k_, "value": v} for k_, v in fc["attributions"].items()])
            attr["direction"] = attr["value"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
            chart = alt.Chart(attr).mark_bar(cornerRadiusEnd=4, height=14).encode(
                y=alt.Y("driver:N", sort=alt.EncodingSortField("value", order="descending"), title=None,
                        axis=alt.Axis(labelOverlap=False, labelLimit=160)),
                x=alt.X("value:Q", title="Contribution to risk (← lowers · raises →)"),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                               range=[INCREASES, DECREASES]), title=None),
                tooltip=[alt.Tooltip("driver:N", title="Driver"), alt.Tooltip("value:Q", title="SHAP", format="+.3f")],
            ).properties(height=28 * len(attr))
            rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
            st.altair_chart(chart + rule, use_container_width=True)
            st.markdown("**Complaint rate by aspect, last 3 months**")
            asp = pd.DataFrame([{"aspect": a, "rate": v["complaint_rate_3m"], "prev": v["complaint_rate_prev6m"],
                                 "z": v["robust_z"], "spike": "⚠️ spike" if v["spike"] else ""}
                                for a, v in m["aspects"].items()])
            st.dataframe(asp, hide_index=True, use_container_width=True, column_config={
                "aspect": "Aspect", "rate": st.column_config.NumberColumn("Now", format="percent"),
                "prev": st.column_config.NumberColumn("Prior 6 mo", format="percent"),
                "z": st.column_config.NumberColumn("Robust z", format="%.1f"), "spike": "Spike"})

    with tab_trend:
        tl = pd.DataFrame(load("timeline", bid))
        tl["date"] = pd.to_datetime(tl["month"])
        tl["rating_roll3"] = (tl["mean_stars"] * tl["n_reviews"]).rolling(3, min_periods=1).sum() / \
            tl["n_reviews"].rolling(3, min_periods=1).sum()
        months = st.slider("Months shown", 12, len(tl), min(36, len(tl)), step=6)
        tl = tl.tail(months)
        base = alt.Chart(tl).encode(x=alt.X("date:T", title=None))
        hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
        st.markdown("**Average star rating** — dots: monthly mean · line: 3-month rolling mean")
        rating = alt.layer(
            base.mark_circle(size=40, color="#86b6ef").encode(y=alt.Y("mean_stars:Q", title="Stars",
                                                                      scale=alt.Scale(domain=[1, 5]))),
            base.mark_line(strokeWidth=2, color="#2a78d6").encode(y="rating_roll3:Q"),
            base.mark_rule(color=MUTED).encode(opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
                                               tooltip=[alt.Tooltip("month:N", title="Month"),
                                                        alt.Tooltip("mean_stars:Q", title="Monthly mean", format=".2f"),
                                                        alt.Tooltip("rating_roll3:Q", title="3-mo mean", format=".2f"),
                                                        alt.Tooltip("n_reviews:Q", title="Reviews")]).add_params(hover),
        ).properties(height=200)
        st.altair_chart(rating, use_container_width=True)
        st.markdown("**Predicted decline risk** — dashed lines: medium / high thresholds")
        risk = alt.layer(
            base.mark_area(color="#d03b3b", opacity=0.12).encode(y=alt.Y("risk:Q", title="Risk", axis=alt.Axis(format="%"))),
            base.mark_line(strokeWidth=2, color="#d03b3b").encode(y="risk:Q"),
            alt.Chart(pd.DataFrame({"y": [0.15, 0.30]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(y="y:Q"),
            base.mark_rule(color=MUTED).encode(opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
                                               tooltip=[alt.Tooltip("month:N", title="Month"),
                                                        alt.Tooltip("risk:Q", title="Risk", format=".0%")]).add_params(hover),
        ).properties(height=160)
        st.altair_chart(risk, use_container_width=True)
        st.markdown("**Complaint rate per aspect** (EWMA) — red dots mark M3 complaint spikes")
        long = pd.concat([tl[["date", "month"]].assign(aspect=a, rate=tl[f"{a}_rate"], z=tl[f"{a}_z"],
                                                       spike=tl[f"{a}_spike"]) for a in ASPECTS])
        small = alt.layer(
            alt.Chart().mark_line(strokeWidth=2).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=4)),
                y=alt.Y("rate:Q", title="Share of reviews", axis=alt.Axis(format="%")),
                color=alt.Color("aspect:N", scale=alt.Scale(domain=ASPECTS, range=list(ASPECT_COLORS.values())), legend=None)),
            alt.Chart().mark_point(filled=True, size=70, color="#d03b3b", stroke="white", strokeWidth=2).encode(
                x="date:T", y="rate:Q",
                tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("aspect:N", title="Aspect"),
                         alt.Tooltip("rate:Q", title="Complaint rate", format=".0%"),
                         alt.Tooltip("z:Q", title="Robust z", format=".1f")]).transform_filter("datum.spike == 1"),
            data=long,
        ).properties(width=170, height=140).facet(column=alt.Column("aspect:N", sort=ASPECTS, title=None))
        st.altair_chart(small)

    with tab_peer:
        st.markdown(f"**Complaint rate vs. peers** — {peers['basis']} ({peers['n_peers']} with recent reviews)")
        pdf = pd.concat([pd.DataFrame({"aspect": a, "who": ["This restaurant", "Peer median"],
                                       "rate": [v["mine"], v["peer_median"]], "pctl": v["percentile"]})
                         for a, v in peers["aspects"].items()])
        chart = alt.Chart(pdf).mark_bar(cornerRadiusEnd=4, height=12).encode(
            y=alt.Y("who:N", title=None, axis=None), x=alt.X("rate:Q", title="Share of reviews complaining",
                                                             axis=alt.Axis(format="%")),
            color=alt.Color("who:N", scale=alt.Scale(domain=["This restaurant", "Peer median"],
                                                     range=["#2a78d6", "#eb6834"]), title=None),
            row=alt.Row("aspect:N", sort=ASPECTS, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=[alt.Tooltip("aspect:N", title="Aspect"), alt.Tooltip("who:N", title="Series"),
                     alt.Tooltip("rate:Q", title="Complaint rate", format=".1%"),
                     alt.Tooltip("pctl:Q", title="Percentile among peers", format=".0f")],
        ).properties(height=34, width=500)
        st.altair_chart(chart)

    with tab_rev:
        c1, c2 = st.columns(2)
        aspect = c1.selectbox("Aspect", ["(all)"] + ASPECTS)
        sentiment = c2.selectbox("Sentiment", ["negative", "positive"], disabled=aspect == "(all)")
        revs = load("reviews", bid, None if aspect == "(all)" else aspect, sentiment, 30)
        for r in revs:
            tags = " ".join(f"`{a}{'+' if v == 1 else '−'}`" for a, v in r["aspects"].items() if v)
            st.markdown(f"**{'★' * int(r['stars'])}{'☆' * (5 - int(r['stars']))}** · {r['date']} · {tags}  \n{r['text']}")
            st.divider()


def page_models():
    ev = load("evaluation")
    st.title("Model performance")
    st.caption("Every number below is produced by the pipeline's evaluation steps (data/work/<source>/outputs).")

    st.header("M2 · Aspect-sentiment classification")
    m2 = ev.get("m2")
    if m2:
        for bench, res in m2["results"].items():
            st.markdown(f"**Benchmark: {bench}** ({m2['benchmarks'][bench]['n']} labelled texts; "
                        f"aspects: {', '.join(m2['benchmarks'][bench]['aspects'])})")
            tbl = pd.DataFrame([{"model": r["model"], "macro-F1": r.get("macro_f1"), "complaint-F1": r.get("complaint_f1"),
                                 "ms / review": r.get("inference_ms_per_review"), "note": r.get("skipped", "")}
                                for r in res])
            st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.caption(f"Selected for production: **{m2['selected_model']}** (by macro-F1 on {m2['selected_on']}).")

    st.header("M3 · Emerging-complaint detection")
    m3 = ev.get("m3")
    if m3:
        st.dataframe(pd.DataFrame([{"detector": k.replace("_", " "), **{kk: vv for kk, vv in v.items()}}
                                   for k, v in m3.items() if isinstance(v, dict)]),
                     hide_index=True, use_container_width=True)
        st.caption(f"Base rate of a decline in the next 3 months: {m3['base_rate']:.1%}. Lift = precision ÷ base rate.")

    st.header("M4 · Rating-decline forecasting (out-of-time test)")
    m4 = ev.get("m4")
    if m4:
        sp = m4["split"]
        st.caption(f"Train {sp['train']} · validation {sp['valid']} · test {sp['test']} "
                   f"({sp['n_test']:,} business-months, {sp['pos_rate_test']:.1%} declines)")
        tbl = pd.DataFrame([{"model": MODEL_LABELS.get(k, k), "AUROC": r["auroc"], "PR-AUC": r["pr_auc"],
                             "precision@10%": r["precision_at_10pct"], "recall@10%": r["recall_at_10pct"],
                             "onsets warned %": r["early_warning"]["detected_pct"],
                             "median lead (months)": r["early_warning"]["median_lead_months"]}
                            for k, r in m4["results"].items()])
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        curves = load("m4_curves")
        pr = pd.concat([pd.DataFrame({"model": MODEL_LABELS.get(k, k), "recall": v["pr"]["recall"],
                                      "precision": v["pr"]["precision"]}) for k, v in curves.items()])
        roc = pd.concat([pd.DataFrame({"model": MODEL_LABELS.get(k, k), "fpr": v["roc"]["fpr"], "tpr": v["roc"]["tpr"]})
                         for k, v in curves.items()])
        dom = [MODEL_LABELS[k] for k in curves]
        col = alt.Color("model:N", scale=alt.Scale(domain=dom, range=MODEL_COLORS[:len(dom)]), title=None,
                        legend=alt.Legend(orient="top", columns=2))
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(pr).mark_line(strokeWidth=2).encode(
            x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision"), color=col,
            tooltip=["model:N", alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")]
        ).properties(title="Precision–recall", height=300), use_container_width=True)
        c2.altair_chart(alt.Chart(roc).mark_line(strokeWidth=2).encode(
            x=alt.X("fpr:Q", title="False-positive rate"), y=alt.Y("tpr:Q", title="True-positive rate"), color=col,
            tooltip=["model:N", alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f")]
        ).properties(title="ROC", height=300), use_container_width=True)
        if m4.get("lgbm_top_features_shap"):
            imp = pd.DataFrame(list(m4["lgbm_top_features_shap"].items()), columns=["feature", "mean |SHAP|"])
            st.altair_chart(alt.Chart(imp).mark_bar(color="#2a78d6", cornerRadiusEnd=4, height=12).encode(
                y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("mean |SHAP|:Q"),
                tooltip=["feature:N", alt.Tooltip("mean |SHAP|:Q", format=".4f")]
            ).properties(title="Top features (LightGBM, mean |SHAP| on test)", height=320), use_container_width=True)

    st.header("M5 · Grounded action plans")
    m5, fb = ev.get("m5"), ev.get("feedback", {})
    if m5:
        k = st.columns(4)
        k[0].metric("Plans generated", m5["plans"])
        k[1].metric("Claims verified", f"{m5['claims_verified']}/{m5['claims_total']}")
        k[2].metric("Groundedness", pct(m5["groundedness"]), help="Target ≥ 90%")
        k[3].metric("Owner usefulness", f"{fb.get('mean_usefulness') or '–'} / 5", help=f"{fb.get('n', 0)} ratings; target ≥ 4")


def page_about():
    st.title("About Radview")
    st.markdown("""
Radview warns restaurant owners **before** their rating falls and tells them, with review evidence,
**which problem to fix first**. It is a six-module pipeline deployed as Docker containers on Kubernetes.
""")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#52514e"];
  data [label="Yelp Open Dataset\\nSemEval-2014 ABSA", fillcolor="#e8f0fb"];
  m1 [label="M1 Ingestion\\nclean · dedupe · parquet", fillcolor="#e8f5ec"];
  m2 [label="M2 Aspect classifier\\nDistilBERT (PyTorch)", fillcolor="#fdf1e3"];
  m3 [label="M3 Trend detector\\nEWMA + robust z", fillcolor="#fdf1e3"];
  m4 [label="M4 Decline forecaster\\nGRU (PyTorch) + SHAP", fillcolor="#fdf1e3"];
  m5 [label="M5 Agent (LangGraph)\\ntools → draft → verify", fillcolor="#f1e8f8"];
  m6 [label="M6 FastAPI + Streamlit", fillcolor="#e3f3f3"];
  cron [label="Weekly CronJob", fillcolor="#e8f5ec"];
  data -> m1 -> m2 -> m3 -> m4 -> m5 -> m6; m2 -> m4; m2 -> m5 [label="evidence"]; cron -> m1; }
""")
    st.markdown("""
| Module | What it does |
|---|---|
| **M1** | Streams the 5 GB Yelp review file, keeps restaurants, cleans & de-duplicates, writes parquet. |
| **M2** | Tags each review's sentiment toward food, service, cleanliness, price and ambience. Trained by weak supervision; evaluated on SemEval-2014. |
| **M3** | Flags aspects whose complaint rate jumps above the restaurant's own history (robust z-score). |
| **M4** | Predicts a significant ≥0.3★ drop over the next 3 months; a PyTorch GRU reads 12 months of signals; SHAP explains each score. |
| **M5** | A LangGraph agent ranks aspects, retrieves evidence reviews, drafts a plan (Claude or deterministic), and a verifier rejects any claim not backed by a cited review. |
| **M6** | REST API + this dashboard, containerised and orchestrated on Kubernetes with a weekly re-scoring CronJob. |
""")
    st.caption(f"Data client mode: {client().mode}")


PAGES = {
    "portfolio": st.Page(page_portfolio, title="Portfolio", icon="📋", url_path="portfolio", default=True),
    "restaurant": st.Page(page_restaurant, title="Restaurant", icon="🍽️", url_path="restaurant"),
    "models": st.Page(page_models, title="Model performance", icon="🧪", url_path="models"),
    "about": st.Page(page_about, title="About", icon="ℹ️", url_path="about"),
}
st.navigation(list(PAGES.values())).run()
