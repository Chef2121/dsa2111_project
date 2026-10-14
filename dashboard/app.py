"""M6 - Radview owner dashboard (Streamlit).

Pages
    Portfolio          every monitored restaurant ranked by decline risk (for chains / analysts)
    Restaurant         one restaurant, in the order an owner asks:
                         Am I in trouble? -> Is my rating moving? -> What are people complaining
                         about (vs. the market)? -> Is it getting worse? -> What do I do first?
    Alerts             live complaint spikes across all restaurants
    Try a review       paste reviews, see aspect sentiment and the complaint sentences (M2 live)
    Model performance  M2-M5 evaluation results and model comparisons
    About              architecture

Design rules: one categorical colour per aspect (never by rank), status colours only for
risk state and always with an icon + label, every chart has a table view, light and dark
palettes, missing months are gaps rather than interpolated.

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
st.set_page_config(page_title="Radview", page_icon="📡", layout="wide")

# ---------------------------------------------------------------- palette (light / dark)
IS_DARK = (st.context.theme.type or "light") == "dark"
if IS_DARK:
    SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
    INK, MUTED, GRID, SOFT = "#ffffff", "#c3c2b7", "#3a3a37", "#2a2a28"
else:
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    INK, MUTED, GRID, SOFT = "#0b0b0b", "#52514e", "#e6e5e0", "#f0efec"
ASPECT_COLORS = dict(zip(ASPECTS, SERIES))          # identity: fixed order, never cycled
STATUS = {"high": ("#d03b3b", "🔴 High"), "medium": ("#ec835a", "🟠 Medium"), "low": ("#0ca30c", "🟢 Low")}
SEVERITY = {"critical": "🔴 critical", "serious": "🟠 serious", "warning": "🟡 warning"}
INCREASES, DECREASES = "#e34948", SERIES[0]          # diverging poles for SHAP direction
MODEL_LABELS = {"majority": "Majority (base rate)", "current_negativity": "Current negativity (no model)",
                "rating_logreg": "Rating trend · LogReg", "rating_lgbm": "Rating trend · LightGBM",
                "sentiment_lgbm": "+ Overall sentiment · LightGBM", "radview_lgbm": "Radview features · LightGBM",
                "radview_gru": "Radview · PyTorch GRU"}
EFFORT_LABEL = {0.6: "quick fix", 0.8: "low effort", 0.9: "moderate effort", 1.0: "moderate effort",
                1.3: "bigger investment"}


@alt.theme.register("radview", enable=True)
def _theme():
    """Recessive axes & grid; text in ink colours, never series colours."""
    return {"config": {"background": "transparent", "view": {"stroke": None}, "font": "sans-serif",
                       "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": GRID, "domainColor": GRID,
                                "tickColor": GRID, "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": "normal"},
                       "legend": {"labelColor": INK, "titleColor": MUTED, "orient": "top", "labelFontSize": 11},
                       "header": {"labelColor": INK, "titleColor": MUTED},
                       "title": {"color": INK, "fontSize": 13, "anchor": "start"}}}



@st.cache_resource
def client() -> Client:
    return Client()


@st.cache_data(ttl=600, show_spinner=False)
def load(kind: str, *args):
    return getattr(client(), kind)(*args)


def pct(x, digits: int = 0) -> str:
    return "–" if x is None else f"{x:.{digits}%}"


def status_badge(level: str) -> str:
    color, label = STATUS[level]
    return (f"<span style='background:{color}26;border:1px solid {color};border-radius:12px;"
            f"padding:2px 10px;font-size:0.85rem;white-space:nowrap'>{label} risk</span>")


def chart_or_table(chart, table: pd.DataFrame, name: str, **chart_kwargs):
    """Every chart has a table view (accessibility + exact numbers)."""
    if st.toggle("Table", key=f"tbl-{name}"):
        st.dataframe(table, hide_index=True, use_container_width=True)
        return None
    return st.altair_chart(chart, use_container_width=True, **chart_kwargs)


# ================================================================== Portfolio
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
                                format_func=lambda lv: STATUS[lv][1])
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
            level=view["risk_level"].map(lambda lv: STATUS[lv][1]),
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
        hist = alt.Chart(view).mark_bar(color=SERIES[0], cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("risk:Q", bin=alt.Bin(step=0.05), title="Decline risk", axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", title="Restaurants"),
            tooltip=[alt.Tooltip("risk:Q", bin=alt.Bin(step=0.05), title="Risk band", format=".0%"),
                     alt.Tooltip("count():Q", title="Restaurants")]).properties(height=200)
        chart_or_table(hist, view[["name", "risk"]].sort_values("risk", ascending=False), "hist")
        st.subheader("Main risk driver")
        drivers = view["top_aspect"].value_counts().rename_axis("aspect").reset_index(name="restaurants")
        bars = alt.Chart(drivers).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("aspect:N", sort="-x", title=None),
            x=alt.X("restaurants:Q", title="Restaurants"),
            color=alt.Color("aspect:N", scale=alt.Scale(domain=ASPECTS, range=SERIES), legend=None),
            tooltip=["aspect:N", "restaurants:Q"]).properties(height=170)
        chart_or_table(bars + bars.mark_text(align="left", dx=4, color=MUTED).encode(text="restaurants:Q"),
                       drivers, "drivers")


# ================================================================== Restaurant
def _verdict(info, m, fc) -> str:
    """One plain-English sentence answering 'am I in trouble, and why?'."""
    top = max(ASPECTS, key=lambda a: fc.get("counterfactual_risk_drop", {}).get(a, 0) + fc["attributions"].get(a, 0))
    am = m["aspects"][top]
    market = (f", against a market median of {pct(am['market_median_rate'])}"
              if am.get("market_median_rate") is not None else "")
    return (f"**{info['name']}** has a **{pct(fc['risk'])}** chance its rating falls by 0.3★ or more over the next "
            f"3 months. The biggest driver is **{top}** — {pct(am['complaint_rate_3m'])} of recent reviews complain "
            f"about it{market}.")


def _plan_section(bid: str):
    with st.spinner("Running the agent…"):
        plan = load("plan", bid)
    v = plan.get("verification", {})
    head = st.columns([5, 1])
    planner = plan.get("planner", "rule_based")
    head[0].markdown(f"### 🧭 What to do first &nbsp; <span style='font-size:0.75rem;border:1px solid {MUTED};"
                     f"border-radius:8px;padding:1px 8px'>{'CLAUDE AGENT' if planner == 'claude' else 'RULE-BASED'}</span>",
                     unsafe_allow_html=True)
    if head[1].button("↻ Re-run", help="Re-run the agent (uses Claude if an API key is configured)"):
        load.clear()
        client().plan(bid, refresh=True)
        st.rerun()
    st.info(plan["summary"])
    st.caption(f"✅ {v.get('claims_verified', 0)}/{v.get('claims_total', 0)} claims verified against your reviews · "
               f"{len(plan.get('tool_calls', []))} tool calls · generated {plan.get('generated_at', '')}")
    if not plan["items"]:
        st.success("No pressing issues found — nothing to fix right now.")
    for it in plan["items"]:
        with st.container(border=True):
            color = ASPECT_COLORS.get(it["aspect"], MUTED)
            badges = []
            if it.get("star_cost"):
                badges.append(f"≈{it['star_cost']:.2f}★ cost vs market")
            if it.get("risk_drop_if_fixed"):
                badges.append(f"−{it['risk_drop_if_fixed']:.0%} risk if fixed")
            badges.append(EFFORT_LABEL.get(it.get("effort"), ""))
            badge_html = " ".join(f"<span style='font-size:0.75rem;background:{SOFT};border-radius:8px;padding:2px 8px;"
                                  f"margin-left:4px'>{b}</span>" for b in badges if b)
            st.markdown(f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};"
                        f"margin-right:6px'></span>**{it['rank']} · {it['aspect'].capitalize()}** — {it['headline']} "
                        f"{badge_html}", unsafe_allow_html=True)
            if it["actions"]:
                st.markdown(f"**Do this:** {it['actions'][0]}")
            ev, act = st.columns([3, 2])
            with ev:
                for c in it["claims"]:
                    if c.get("type") == "review":
                        st.markdown(f"> {c['text']}  \n> ✅ verified · `{', '.join(c.get('review_ids', []))}`")
                    else:
                        st.markdown(f"📊 {c['text']} ✅")
            with act:
                if len(it["actions"]) > 1:
                    st.markdown("**Also consider**")
                    for a in it["actions"][1:]:
                        st.markdown(f"- {a}")
                st.caption(it["why"])
    rejected = plan.get("rejected_claims") or []
    with st.expander(f"Verifier log — {len(rejected)} claim(s) rejected before you saw them"):
        if rejected:
            st.dataframe(pd.DataFrame(rejected), hide_index=True, use_container_width=True)
        else:
            st.write("Every drafted claim passed citation, quotation, relevance and statistic checks.")
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
    names = dict(zip(rows["business_id"],
                     rows["name"] + " — " + (rows["risk"] * 100).round().astype(int).astype(str) + "% risk"))
    default = st.session_state.get("bid", ids[0])
    bid = st.selectbox("Restaurant", ids, index=ids.index(default) if default in ids else 0, format_func=names.get)
    if bid != st.session_state.get("bid"):
        st.session_state.pop("issue", None)
    st.session_state["bid"] = bid
    d = load("restaurant", bid)
    info, m, fc, peers = d["info"], d["metrics"], d["forecast"], d["peers"]
    tl = pd.DataFrame(load("timeline", bid))
    tl["date"] = pd.to_datetime(tl["month"])

    # --- Am I in trouble? -------------------------------------------------------
    with st.container(border=True):
        st.markdown(f"{status_badge(fc['risk_level'])} &nbsp; <span style='color:{MUTED};font-size:0.85rem'>"
                    f"{info['cuisine']} · {info['address']}, {info['city']} · analysis for {m['month']}</span>",
                    unsafe_allow_html=True)
        st.markdown(_verdict(info, m, fc))
    spikes = [a for a, v in m["aspects"].items() if v["spike"]]
    k = st.columns(4)
    k[0].metric("Chance rating falls", pct(fc["risk"]), help="next 3 months, ≥0.3★, calibrated probability")
    delta = None if m["rating_3m"] is None or m["rating_6m"] is None else round(m["rating_3m"] - m["rating_6m"], 2)
    k[1].metric("Average rating (3 mo)", f"{m['rating_3m']:.2f} ★" if m["rating_3m"] else "–",
                delta=f"{delta:+.2f} vs 6 mo" if delta is not None else None)
    k[2].metric("Reviews (3 mo)", m["reviews_3m"])
    k[3].metric("Live alerts", len(spikes), help=", ".join(spikes) if spikes else "nothing spiking right now")

    # --- Is my rating moving? / What are people complaining about? -------------
    left, right = st.columns(2)
    with left:
        st.markdown("**Average rating by month** — gaps are months with no reviews")
        shown = tl.tail(36)
        hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
        base = alt.Chart(shown).encode(x=alt.X("date:T", title=None))
        rating = alt.layer(
            base.mark_line(strokeWidth=2, color=SERIES[0], point=alt.OverlayMarkDef(size=30, color=SERIES[0])).encode(
                y=alt.Y("mean_stars:Q", title="Stars", scale=alt.Scale(domain=[1, 5]))),
            base.mark_rule(color=MUTED).encode(
                opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
                tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("mean_stars:Q", title="Mean", format=".2f"),
                         alt.Tooltip("n_reviews:Q", title="Reviews")]).add_params(hover),
        ).properties(height=260)
        chart_or_table(rating, shown[["month", "mean_stars", "n_reviews"]], "rating")
    with right:
        st.markdown("**What people complain about** — your rate vs. the market median (│)")
        comp = pd.DataFrame([{"aspect": a, "rate": v["complaint_rate_3m"], "market": v.get("market_median_rate"),
                              "z": v["robust_z"]} for a, v in m["aspects"].items()])
        bars = alt.Chart(comp).mark_bar(cornerRadiusEnd=4, height=22).encode(
            y=alt.Y("aspect:N", sort=ASPECTS, title=None, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("rate:Q", title="Share of recent reviews complaining", axis=alt.Axis(format="%")),
            color=alt.Color("aspect:N", scale=alt.Scale(domain=ASPECTS, range=SERIES), legend=None),
            tooltip=[alt.Tooltip("aspect:N", title="Issue"), alt.Tooltip("rate:Q", title="You", format=".1%"),
                     alt.Tooltip("market:Q", title="Market median", format=".1%"),
                     alt.Tooltip("z:Q", title="Robust z", format=".1f")])
        ticks = alt.Chart(comp).mark_tick(color=INK, thickness=2, size=28).encode(
            y=alt.Y("aspect:N", sort=ASPECTS), x="market:Q")
        chart_or_table(alt.layer(bars, ticks).properties(height=230), comp, "complaints")
        worst = max(ASPECTS, key=lambda a: m["aspects"][a]["complaint_rate_3m"])
        current = st.session_state.get("issue") if st.session_state.get("issue") in ASPECTS else worst
        chosen = st.pills("Show the trend for", ASPECTS, default=current, key=f"issue-pills-{bid}")
        st.session_state["issue"] = chosen or current

    # --- Is it getting worse? ---------------------------------------------------------
    issue = st.session_state["issue"]
    st.markdown(f"**{issue.capitalize()} complaints over time** — monthly share (EWMA) vs. the market median; "
                f"red dots are M3 complaint spikes")
    sub = tl.tail(36)
    series = ["monthly (EWMA)", "market median (3 mo)"]
    long = pd.concat([
        pd.DataFrame({"date": sub["date"], "month": sub["month"], "series": series[0], "value": sub[f"{issue}_rate"]}),
        pd.DataFrame({"date": sub["date"], "month": sub["month"], "series": series[1], "value": sub[f"{issue}_market_rate"]}),
    ])
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("value:Q", title="Share of reviews", axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(domain=series, range=[ASPECT_COLORS[issue], MUTED]), title=None,
                        legend=alt.Legend(orient="top", symbolStrokeWidth=3)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=series, range=[[1, 0], [4, 4]]), legend=None),
        tooltip=["month:N", "series:N", alt.Tooltip("value:Q", format=".1%")])
    dots = alt.Chart(sub[sub[f"{issue}_spike"] == 1]).mark_point(
        filled=True, size=90, color="#d03b3b", stroke="white", strokeWidth=2).encode(
        x="date:T", y=f"{issue}_rate:Q",
        tooltip=[alt.Tooltip("month:N", title="Spike"), alt.Tooltip(f"{issue}_z:Q", title="Robust z", format=".1f")])
    chart_or_table(alt.layer(lines, dots).properties(height=220),
                   sub[["month", f"{issue}_rate", f"{issue}_neg_rate_3m", f"{issue}_market_rate", f"{issue}_z",
                        f"{issue}_spike"]], "issue")

    # --- What do I do on Monday? --------------------------------------------------------
    _plan_section(bid)

    # --- Deeper dives ---------------------------------------------------------------
    tab_drv, tab_risk, tab_peer, tab_rev = st.tabs(["🔍 Why at risk", "📈 Risk history", "👥 Peers", "💬 Reviews"])
    with tab_drv:
        st.caption("Two independent explanations of the same forecast. SHAP splits the score across inputs; the "
                   "counterfactual asks the model how much the risk drops if one aspect looked like the market norm.")
        c1, c2 = st.columns(2)
        attr = pd.DataFrame([{"driver": k_, "value": v} for k_, v in fc["attributions"].items()])
        attr["direction"] = attr["value"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
        shap_chart = alt.Chart(attr).mark_bar(cornerRadiusEnd=4, height=14).encode(
            y=alt.Y("driver:N", sort=alt.EncodingSortField("value", order="descending"), title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=160)),
            x=alt.X("value:Q", title="SHAP contribution (logit)"),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                           range=[INCREASES, DECREASES]), title=None),
            tooltip=["driver:N", alt.Tooltip("value:Q", format="+.3f")]).properties(height=28 * len(attr), title="SHAP")
        with c1:
            chart_or_table(shap_chart, attr, "shap")
        cf = pd.DataFrame([{"aspect": a, "risk drop if fixed": fc.get("counterfactual_risk_drop", {}).get(a, 0),
                            "star cost": fc.get("star_cost", {}).get(a, 0)} for a in ASPECTS])
        cf_chart = alt.Chart(cf).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("aspect:N", sort="-x", title=None),
            x=alt.X("risk drop if fixed:Q", title="Risk reduction if at market norm", axis=alt.Axis(format="%")),
            color=alt.Color("aspect:N", scale=alt.Scale(domain=ASPECTS, range=SERIES), legend=None),
            tooltip=["aspect:N", alt.Tooltip("risk drop if fixed:Q", format=".1%"),
                     alt.Tooltip("star cost:Q", title="Star cost", format=".2f")]).properties(height=170, title="Counterfactual")
        with c2:
            chart_or_table(cf_chart, cf, "cf")
    with tab_risk:
        risk = alt.layer(
            alt.Chart(tl).mark_area(color="#d03b3b", opacity=0.12).encode(
                x=alt.X("date:T", title=None), y=alt.Y("risk:Q", title="Risk", axis=alt.Axis(format="%"))),
            alt.Chart(tl).mark_line(strokeWidth=2, color="#d03b3b").encode(
                x="date:T", y="risk:Q", tooltip=["month:N", alt.Tooltip("risk:Q", format=".0%")]),
            alt.Chart(pd.DataFrame({"y": [0.15, 0.30]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(y="y:Q"),
        ).properties(height=220)
        st.caption("Predicted decline risk each month; dashed lines mark the medium / high thresholds.")
        chart_or_table(risk, tl[["month", "risk", "decline"]], "risk")
    with tab_peer:
        st.markdown(f"**Complaint rate vs. peers** — {peers['basis']} ({peers['n_peers']} with recent reviews)")
        pdf = pd.concat([pd.DataFrame({"aspect": a, "who": ["This restaurant", "Peer median"],
                                       "rate": [v["mine"], v["peer_median"]], "percentile": v["percentile"]})
                         for a, v in peers["aspects"].items()])
        peer_chart = alt.Chart(pdf).mark_bar(cornerRadiusEnd=4, height=12).encode(
            y=alt.Y("who:N", title=None, axis=None),
            x=alt.X("rate:Q", title="Share of reviews complaining", axis=alt.Axis(format="%")),
            color=alt.Color("who:N", scale=alt.Scale(domain=["This restaurant", "Peer median"],
                                                     range=[SERIES[0], SERIES[1]]), title=None),
            row=alt.Row("aspect:N", sort=ASPECTS, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=["aspect:N", "who:N", alt.Tooltip("rate:Q", format=".1%"),
                     alt.Tooltip("percentile:Q", title="Percentile among peers", format=".0f")],
        ).properties(height=34, width=500)
        if st.toggle("Table", key="tbl-peers"):
            st.dataframe(pdf, hide_index=True, use_container_width=True)
        else:
            st.altair_chart(peer_chart)
    with tab_rev:
        c1, c2 = st.columns(2)
        aspect = c1.selectbox("Aspect", ["(all)"] + ASPECTS)
        sentiment = c2.selectbox("Sentiment", ["negative", "positive"], disabled=aspect == "(all)")
        for r in load("reviews", bid, None if aspect == "(all)" else aspect, sentiment, 30):
            tags = " ".join(f"`{a}{'+' if v == 1 else '−'}`" for a, v in r["aspects"].items() if v)
            st.markdown(f"**{'★' * int(r['stars'])}{'☆' * (5 - int(r['stars']))}** · {r['date']} · {tags}  \n{r['text']}")
            st.divider()


# ================================================================== Alerts
def page_alerts():
    st.title("🚨 Live complaint alerts")
    st.caption("Aspects whose complaint rate broke away from the restaurant's own 12-month baseline this month "
               "(M3 robust z-score). Click a row to open the restaurant.")
    rows = pd.DataFrame(load("alerts"))
    if rows.empty:
        st.success("Nothing is spiking right now.")
        return
    sev = st.segmented_control("Severity", list(SEVERITY), selection_mode="multi", default=list(SEVERITY),
                               format_func=SEVERITY.get)
    rows = rows[rows["severity"].isin(sev or list(SEVERITY))]
    table = rows.assign(level=rows["severity"].map(SEVERITY))[
        ["name", "aspect", "level", "robust_z", "complaint_rate_3m", "market_median_rate", "risk", "business_id"]]
    event = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                         column_config={
                             "name": "Restaurant", "aspect": "Issue", "level": "Severity",
                             "robust_z": st.column_config.NumberColumn("Robust z", format="%.1f"),
                             "complaint_rate_3m": st.column_config.NumberColumn("Complaint rate", format="percent"),
                             "market_median_rate": st.column_config.NumberColumn("Market median", format="percent"),
                             "risk": st.column_config.ProgressColumn("Decline risk", min_value=0, max_value=1,
                                                                    format="percent"),
                             "business_id": None})
    if event.selection.rows:
        st.session_state["bid"] = table.iloc[event.selection.rows[0]]["business_id"]
        st.session_state["issue"] = table.iloc[event.selection.rows[0]]["aspect"]
        st.switch_page(PAGES["restaurant"])


# ================================================================== Try a review
def page_analyse():
    st.title("🔬 Try a review")
    st.caption("Paste one review per line. The production aspect model (M2) labels each aspect and highlights the "
               "sentence that carries each complaint — the same check the agent's verifier uses.")
    text = st.text_area("Reviews", height=140, value=(
        "The pasta was cold and the sauce tasted like it came from a jar. Our server was friendly though.\n"
        "Lovely patio and great prices, but the bathroom was filthy."))
    if st.button("Analyse", type="primary"):
        texts = [t.strip() for t in text.splitlines() if t.strip()][:50]
        for res in client().analyse(texts):
            with st.container(border=True):
                body = res["text"]
                for a, sent in res["complaint_sentences"].items():
                    body = body.replace(sent, f"<mark style='background:{ASPECT_COLORS[a]}40;color:inherit'>{sent}</mark>")
                st.markdown(body, unsafe_allow_html=True)
                chips = [f"{'👎' if lab == 'negative' else '👍'} {a} ({res['confidence'][a]:.0%})"
                         for a, lab in res["aspects"].items() if lab != "not mentioned"]
                st.caption(" · ".join(chips) or "no aspects detected")


# ================================================================== Model performance
def page_models():
    ev = load("evaluation")
    st.title("🧪 Model performance")
    st.caption("Every number below is produced by the pipeline's evaluation steps (data/work/<source>/outputs).")

    st.header("M2 · Aspect-sentiment classification")
    m2 = ev.get("m2")
    if m2:
        for bench, res in m2["results"].items():
            st.markdown(f"**Benchmark: {bench}** ({m2['benchmarks'][bench]['n']} labelled texts; "
                        f"aspects: {', '.join(m2['benchmarks'][bench]['aspects'])})")
            st.dataframe(pd.DataFrame([{"model": r["model"], "macro-F1": r.get("macro_f1"),
                                        "complaint-F1": r.get("complaint_f1"),
                                        "ms / review": r.get("inference_ms_per_review"), "note": r.get("skipped", "")}
                                       for r in res]), hide_index=True, use_container_width=True)
        st.caption(f"Selected for production: **{m2['selected_model']}** (by macro-F1 on {m2['selected_on']}).")

    st.header("M3 · Emerging-complaint detection")
    m3 = ev.get("m3")
    if m3:
        st.dataframe(pd.DataFrame([{"detector": k.replace("_", " "), **v} for k, v in m3.items() if isinstance(v, dict)]),
                     hide_index=True, use_container_width=True)
        st.caption(f"Base rate of a decline in the next 3 months: {m3['base_rate']:.1%}. Lift = precision ÷ base rate.")

    st.header("M4 · Rating-decline forecasting (out-of-time test)")
    m4 = ev.get("m4")
    if m4:
        sp = m4["split"]
        st.caption(f"Train {sp['train']} · validation {sp['valid']} · test {sp['test']} "
                   f"({sp['n_test']:,} business-months, {sp['pos_rate_test']:.1%} declines). All models are "
                   f"Platt-calibrated on the validation months.")
        st.dataframe(pd.DataFrame([{
            "model": MODEL_LABELS.get(k, k), "AUROC": r["auroc"], "PR-AUC": r["pr_auc"], "Brier ↓": r.get("brier"),
            "precision@10%": r["precision_at_10pct"], "recall@10%": r["recall_at_10pct"],
            "onsets warned %": (r.get("early_warning") or {}).get("detected_pct"),
            "median lead (mo)": (r.get("early_warning") or {}).get("median_lead_months")}
            for k, r in m4["results"].items()]), hide_index=True, use_container_width=True)
        if m4.get("injected_episodes"):
            st.markdown("**Injected deterioration episodes (synthetic ground truth)** — caught at all, and caught "
                        "*before* the 3-month rating visibly dropped")
            st.dataframe(pd.DataFrame([{"model": MODEL_LABELS.get(k, k), **v} for k, v in m4["injected_episodes"].items()]),
                         hide_index=True, use_container_width=True)
        curves = load("m4_curves")
        if curves:
            dom = [MODEL_LABELS.get(k, k) for k in curves]
            col = alt.Color("model:N", scale=alt.Scale(domain=dom, range=(SERIES + ["#4a3aa7", "#008300"])[:len(dom)]),
                            title=None, legend=alt.Legend(orient="top", columns=3))
            pr = pd.concat([pd.DataFrame({"model": MODEL_LABELS.get(k, k), "recall": v["pr"]["recall"],
                                          "precision": v["pr"]["precision"]}) for k, v in curves.items()])
            roc = pd.concat([pd.DataFrame({"model": MODEL_LABELS.get(k, k), "fpr": v["roc"]["fpr"], "tpr": v["roc"]["tpr"]})
                             for k, v in curves.items()])
            c1, c2 = st.columns(2)
            with c1:
                chart_or_table(alt.Chart(pr).mark_line(strokeWidth=2).encode(
                    x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision"), color=col,
                    tooltip=["model:N", alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")]
                ).properties(title="Precision–recall", height=320), pr, "pr")
            with c2:
                chart_or_table(alt.Chart(roc).mark_line(strokeWidth=2).encode(
                    x=alt.X("fpr:Q", title="False-positive rate"), y=alt.Y("tpr:Q", title="True-positive rate"),
                    color=col, tooltip=["model:N", alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f")]
                ).properties(title="ROC", height=320), roc, "roc")
        agr = m4.get("attribution_agreement")
        if agr:
            a1, a2, a3 = st.columns(3)
            a1.metric("SHAP vs counterfactual: same top aspect", pct(agr["all"].get("top_aspect_agreement")),
                      help=f"over {agr['all'].get('n')} restaurants")
            a2.metric("…among at-risk restaurants", pct(agr["at_risk"].get("top_aspect_agreement")),
                      help=f"over {agr['at_risk'].get('n')} restaurants")
            a3.metric("Mean rank correlation (Spearman)", f"{agr['all'].get('mean_spearman') or 0:.2f}")
            st.caption("Star-impact model (ridge, monthly stars ~ complaint rates; stars per unit complaint rate): " +
                       ", ".join(f"{a} {c:+.2f}" for a, c in agr["star_impact_coefficients"].items()))
        if m4.get("lgbm_top_features_shap"):
            imp = pd.DataFrame(list(m4["lgbm_top_features_shap"].items()), columns=["feature", "mean |SHAP|"])
            chart_or_table(alt.Chart(imp).mark_bar(color=SERIES[0], cornerRadiusEnd=4, height=12).encode(
                y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
                x=alt.X("mean |SHAP|:Q"), tooltip=["feature:N", alt.Tooltip("mean |SHAP|:Q", format=".4f")]
            ).properties(title="Top features (LightGBM, mean |SHAP| on test)", height=24 * len(imp)), imp, "imp")

    st.header("M5 · Grounded action plans")
    m5, fb, comp = ev.get("m5"), ev.get("feedback", {}), ev.get("m5_comparison")
    if m5:
        k = st.columns(4)
        k[0].metric("Plans generated", m5["plans"])
        k[1].metric("Claims verified", f"{m5['claims_verified']}/{m5['claims_total']}")
        k[2].metric("Groundedness", pct(m5["groundedness"]), help="Target ≥ 90%")
        k[3].metric("Owner usefulness", f"{fb.get('mean_usefulness') or '–'} / 5",
                    help=f"{fb.get('n', 0)} ratings; target ≥ 4")
    if comp:
        st.markdown(f"**Planner comparison** on the {comp['restaurants']} riskiest restaurants — the adversarial "
                    "planner fabricates claims on purpose; the verifier must reject all of them.")
        st.dataframe(pd.DataFrame([{"planner": k, "groundedness": v["groundedness"], "claims": v["claims_total"],
                                    **{f"✗ {a}": n for a, n in v["failures_by_axis"].items()},
                                    "items / plan": v["mean_items_per_plan"], "latency (s)": v["median_latency_s"],
                                    "tool calls": v["mean_tool_calls"], "tokens": v["mean_tokens"]}
                                   for k, v in comp["planners"].items()]), hide_index=True, use_container_width=True)
        if comp.get("note"):
            st.caption(comp["note"])


# ================================================================== About
def page_about():
    st.title("ℹ️ About Radview")
    st.markdown("Radview warns restaurant owners **before** their rating falls and tells them, with review evidence, "
                "**which problem to fix first**. Six modules, deployed as Docker containers on Kubernetes.")
    st.graphviz_chart(f"""
digraph {{ rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="{MUTED}", fontcolor="#0b0b0b"];
  edge [color="{MUTED}", fontcolor="{MUTED}", fontsize=9];
  data [label="Yelp Open Dataset\\nSemEval-2014 ABSA", fillcolor="#e8f0fb"];
  m1 [label="M1 Ingestion\\nstream · clean · dedupe", fillcolor="#e8f5ec"];
  m2 [label="M2 Aspect classifier\\nDistilBERT (PyTorch)", fillcolor="#fdf1e3"];
  m3 [label="M3 Trend detector\\nEWMA + robust z", fillcolor="#fdf1e3"];
  m4 [label="M4 Decline forecaster\\nGRU + SHAP + counterfactual", fillcolor="#fdf1e3"];
  m5 [label="M5 Agent (LangGraph)\\nClaude tools → verify", fillcolor="#f1e8f8"];
  m6 [label="M6 FastAPI + Streamlit", fillcolor="#e3f3f3"];
  cron [label="Weekly CronJob", fillcolor="#e8f5ec"];
  data -> m1 -> m2 -> m3 -> m4 -> m5 -> m6; m2 -> m4; m2 -> m5 [label="evidence"]; cron -> m1; }}
""")
    st.markdown("""
| Module | What it does |
|---|---|
| **M1** | Streams the 5 GB Yelp review file, keeps one city's restaurants, cleans & de-duplicates, writes parquet. |
| **M2** | Fine-tuned DistilBERT tags each review's sentiment toward food, service, cleanliness, price and ambience — from the text, not the star rating. Trained by weak supervision; evaluated on human-labelled SemEval-2014. |
| **M3** | Flags aspects whose complaint rate jumps above the restaurant's own history (robust z-score). |
| **M4** | Predicts a significant ≥0.3★ drop over the next 3 months; a PyTorch GRU reads 12 months of signals incl. market-relative features. Explained twice — SHAP and a counterfactual — plus a star-cost estimate. |
| **M5** | A LangGraph agent: Claude calls the analytics tools itself and submits a structured plan; a mechanical verifier rejects any claim that fails citation, quotation, relevance or statistic checks. A rule-based planner is the fallback and the baseline. |
| **M6** | REST API (`/docs`) + this dashboard; one Docker image; Kubernetes Job, weekly CronJob and Deployments. |
""")
    st.caption(f"Data client mode: {client().mode} · theme: {'dark' if IS_DARK else 'light'}")


PAGES = {
    "portfolio": st.Page(page_portfolio, title="Portfolio", icon="📋", url_path="portfolio", default=True),
    "restaurant": st.Page(page_restaurant, title="Restaurant", icon="🍽️", url_path="restaurant"),
    "alerts": st.Page(page_alerts, title="Alerts", icon="🚨", url_path="alerts"),
    "analyse": st.Page(page_analyse, title="Try a review", icon="🔬", url_path="analyse"),
    "models": st.Page(page_models, title="Model performance", icon="🧪", url_path="models"),
    "about": st.Page(page_about, title="About", icon="ℹ️", url_path="about"),
}
st.navigation(list(PAGES.values())).run()
