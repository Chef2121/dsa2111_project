"""M6 - Radview owner dashboard (Streamlit).

Pages
    Portfolio          every monitored restaurant ranked by decline risk (for chains / analysts)
    Restaurant         one restaurant, in the order an owner asks:
                         Am I in trouble? -> Is my rating moving? -> What are people complaining
                         about (vs. the market)? -> Is it getting worse? -> What do I do first?
    Alerts             live complaint spikes across all restaurants
    Try a review       paste reviews, see aspect sentiment and the complaint sentences (M2 live)
    About              architecture

Design rules: a restrained palette - navy for single-series data, one muted identity colour per
aspect (never by rank), slate for comparisons, and muted red / amber / green only for risk
state (always with a label). Every chart sits in a
card whose "Data" button shows the exact numbers. Light and dark palettes; missing months
are gaps rather than interpolated.

Run:  streamlit run dashboard/app.py
      (set RADVIEW_API_URL=http://localhost:8000 to read from the API instead of in-process)
"""
from __future__ import annotations

import html
import sys
from contextlib import contextmanager
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `streamlit run dashboard/app.py`
from dashboard.client import Client  # noqa: E402

ASPECTS = ["food", "service", "cleanliness", "price", "ambience"]
ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="Radview", page_icon=str(ASSETS / "mark.svg"), layout="wide")

# ---------------------------------------------------------------- palette (light / dark)
IS_DARK = (st.context.theme.type or "light") == "dark"
if IS_DARK:
    INK, MUTED, GRID, SOFT = "#e7e9ee", "#9aa0ab", "#2a2e36", "#1d2027"
    BG, CARD, ACCENT, COMPARE = "#0e1014", "#15181e", "#8ea8ff", "#6b7280"
    STATUS = {"high": ("#f97066", "High"), "medium": ("#f5a35c", "Medium"), "low": ("#47cd89", "Low")}
    # muted, desaturated identity colours: one per aspect, never by rank
    ASPECT_PALETTE = ["#8fa8d8", "#d6a77e", "#6fc2ad", "#b5a0d6", "#e0a4b3"]
else:
    INK, MUTED, GRID, SOFT = "#101828", "#667085", "#eaecf0", "#f2f4f7"
    BG, CARD, ACCENT, COMPARE = "#f8f9fb", "#ffffff", "#1e3a8a", "#98a2b3"
    STATUS = {"high": ("#b42318", "High"), "medium": ("#b54708", "Medium"), "low": ("#067647", "Low")}
    ASPECT_PALETTE = ["#3b5b92", "#b5835a", "#3f8f7b", "#8a6fb0", "#c27c8e"]
ASPECT_COLORS = dict(zip(ASPECTS, ASPECT_PALETTE))   # food steel-blue, service bronze, cleanliness sage, ...
ASPECT_SCALE = alt.Scale(domain=ASPECTS, range=ASPECT_PALETTE)
RISK = STATUS["high"][0]
SEVERITY = {"critical": "Critical", "serious": "Serious", "warning": "Warning"}
INCREASES, DECREASES = RISK, ACCENT                  # diverging poles for SHAP direction
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
EFFORT_LABEL = {0.6: "quick fix", 0.8: "low effort", 0.9: "moderate effort", 1.0: "moderate effort",
                1.3: "bigger investment"}


@alt.theme.register("radview", enable=True)
def _theme():
    """Quiet axes (no domain lines or ticks, faint grid); text in ink colours, never series colours."""
    return {"config": {"background": "transparent", "view": {"stroke": None}, "font": FONT,
                       "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": GRID, "domain": False,
                                "ticks": False, "labelPadding": 6, "titlePadding": 10, "labelFontSize": 11,
                                "titleFontSize": 11, "titleFontWeight": 500},
                       "axisX": {"grid": False, "domain": True, "domainColor": GRID},
                       "legend": {"labelColor": INK, "titleColor": MUTED, "orient": "top", "labelFontSize": 11},
                       "header": {"labelColor": INK, "titleColor": MUTED},
                       "title": {"color": INK, "fontSize": 13, "anchor": "start"}}}


# ---------------------------------------------------------------- page styling
# Streamlit's defaults (Source Sans, red primary colour, rainbow top bar) are replaced with one
# restrained look: Inter, a navy accent, white cards on a soft grey canvas. Colours come from the
# palette above, so light and dark mode stay in step with the charts.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp {{ background: {BG}; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stSidebarNav"] a span:not([data-testid="stIconMaterial"]) {{ font-family: {FONT}; }}
.stApp p, .stApp li {{ color: {INK}; }}
[data-testid="stDecoration"], #MainMenu, footer, [data-testid="stToolbar"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
.stApp h3 {{ font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em; color: {INK}; }}
[data-testid="stCaptionContainer"] p {{ color: {MUTED}; font-size: 0.8rem; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {GRID}; }}
[data-testid="stSidebarNav"] a span {{ font-weight: 500; font-size: 0.9rem; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {SOFT}; }}
.rv-side-foot {{ font-size: 0.75rem; color: {MUTED}; line-height: 1.6; }}
.rv-side-foot b {{ color: {INK}; font-weight: 600; }}

/* cards: every container keyed "card-..." */
[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {GRID}; border-radius: 12px;
    padding: 18px 20px 16px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }}
.rv-card-title {{ font-size: 0.95rem; font-weight: 600; color: {INK}; letter-spacing: -0.005em; }}
.rv-card-sub {{ font-size: 0.8rem; color: {MUTED}; margin-top: 2px; }}

/* metrics */
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {GRID}; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }}
[data-testid="stMetricLabel"] p {{ font-size: 0.75rem; font-weight: 500; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; color: {INK}; }}
[data-testid="stMetricDelta"] {{ font-size: 0.78rem; }}

/* controls in the accent instead of Streamlit red */
.stApp button {{ border-radius: 8px; }}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"]:hover {{ background: {ACCENT};
    border-color: {ACCENT}; color: {CARD}; }}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] span {{ color: {CARD} !important;
    font-weight: 600; }}
[data-testid="stBaseButton-primary"]:hover {{ filter: brightness(1.12); }}
[data-testid="stBaseButton-pillsActive"], [data-testid="stBaseButton-segmented_controlActive"] {{
    background: {SOFT}; border-color: {INK}; }}
[data-testid="stBaseButton-pillsActive"] p, [data-testid="stBaseButton-segmented_controlActive"] p {{
    color: {INK}; font-weight: 600; }}
[data-testid="stPopover"] {{ display: flex; justify-content: flex-end; }}
[data-testid="stPopover"] button {{ padding: 2px 10px; min-height: 30px; white-space: nowrap; }}
[data-testid="stPopover"] button p {{ font-size: 0.78rem; color: {MUTED}; white-space: nowrap; }}
[data-baseweb="tag"] {{ background: {ACCENT} !important; }}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"]:focus-within {{ border-color: {ACCENT} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 20px; border-bottom: 1px solid {GRID}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {INK}; height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 0.85rem; color: {MUTED}; font-weight: 500; }}
.stTabs [aria-selected="true"] p {{ color: {INK}; font-weight: 600; }}
[data-testid="stAlert"] {{ border-radius: 10px; }}
[data-testid="stExpander"] details {{ border-color: {GRID}; border-radius: 10px; background: {CARD}; }}
[data-testid="stMarkdownContainer"] blockquote {{ border-left: 2px solid {GRID}; color: {MUTED}; }}

/* page header and inline elements */
.rv-eyebrow {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED}; }}
.rv-title {{ font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; color: {INK}; line-height: 1.25;
    margin: 4px 0 4px; }}
.rv-sub {{ color: {MUTED}; font-size: 0.88rem; margin: 0 0 20px; }}
.rv-badge {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid {GRID}; border-radius: 999px;
    padding: 2px 10px; font-size: 0.78rem; font-weight: 500; color: {INK}; background: {CARD}; white-space: nowrap; }}
.rv-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--c); display: inline-block; }}
.rv-tag {{ font-size: 0.72rem; font-weight: 500; background: {SOFT}; color: {MUTED}; border-radius: 6px;
    padding: 2px 8px; margin-left: 4px; white-space: nowrap; }}
.rv-ok {{ font-size: 0.68rem; font-weight: 600; color: {STATUS['low'][0]}; letter-spacing: 0.06em; }}
.rv-rank {{ display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px;
    border-radius: 6px; background: {INK}; color: {CARD}; font-size: 0.75rem; font-weight: 600; margin-right: 8px; }}
.rv-verdict {{ font-size: 1.02rem; line-height: 1.6; color: {INK}; margin-top: 10px; }}
.rv-meta {{ color: {MUTED}; font-size: 0.8rem; margin-left: 8px; }}
</style>
""", unsafe_allow_html=True)
st.logo(str(ASSETS / ("logo-dark.svg" if IS_DARK else "logo.svg")), icon_image=str(ASSETS / "mark.svg"))


@st.cache_resource
def client() -> Client:
    """One data client per server process (API over HTTP, or the service layer in-process)."""
    return Client()


@st.cache_data(ttl=600, show_spinner=False)
def load(kind: str, *args):
    """Cached call of ``Client.<kind>(*args)``; entries expire after 10 minutes so a rescore shows up."""
    return getattr(client(), kind)(*args)


def pct(x, digits: int = 0) -> str:
    """Fraction as a percentage, or an en dash for missing values."""
    return "–" if x is None else f"{x:.{digits}%}"


def md(text: str) -> str:
    """Review/plan text for st.markdown: escape '$' so prices like "$28 ... $13" are not typeset as LaTeX."""
    return text.replace("$", "\\$")


def header(title: str, subtitle: str = "", eyebrow: str = "Radview"):
    """Page heading: small eyebrow, title, muted subtitle (HTML, so callers escape user data)."""
    st.markdown(f"<div class='rv-eyebrow'>{eyebrow}</div><div class='rv-title'>{title}</div>"
                + (f"<p class='rv-sub'>{subtitle}</p>" if subtitle else ""), unsafe_allow_html=True)


def status_badge(level: str, suffix: str = " risk") -> str:
    """Pill with a coloured dot and a text label for a risk tier (colour is never the only cue)."""
    color, label = STATUS[level]
    return f"<span class='rv-badge'><span class='rv-dot' style='--c:{color}'></span>{label}{suffix}</span>"


@contextmanager
def card(key: str, title: str = "", subtitle: str = "", data: pd.DataFrame | None = None):
    """A white card with a title row; ``data`` adds a "Data" button showing the exact numbers behind the chart."""
    with st.container(key=f"card-{key}"):
        if title or data is not None:
            left, right = st.columns([4, 1], vertical_alignment="center")
            left.markdown(f"<div class='rv-card-title'>{title}</div>"
                          + (f"<div class='rv-card-sub'>{subtitle}</div>" if subtitle else ""), unsafe_allow_html=True)
            if data is not None:
                with right.popover("Data", icon=":material/table_rows:"):
                    st.dataframe(data, hide_index=True, use_container_width=True)
        yield


def chart(c, **kwargs):
    """Render an Altair chart at the full width of its card."""
    return st.altair_chart(c, use_container_width=True, **kwargs)


# ================================================================== Portfolio
def page_portfolio():
    """All restaurants: filters, headline counts, the risk ranking (click to open) and two summary charts."""
    meta = load("meta")
    rows = pd.DataFrame(load("restaurants"))
    header("Portfolio", f"{meta['restaurants']} restaurants ranked by the chance their rating falls in the next "
                        f"three months · data through {meta['latest_month']}")

    c1, c2, c3 = st.columns([2, 2, 3])
    cuisines = sorted(rows["cuisine"].dropna().unique())
    pick_cuisine = c1.multiselect("Cuisine", cuisines, placeholder="All cuisines")
    pick_level = c2.multiselect("Risk level", ["high", "medium", "low"], placeholder="All levels",
                                format_func=lambda lv: STATUS[lv][1])
    query = c3.text_input("Search", placeholder="Restaurant name")
    view = rows.copy()
    if pick_cuisine:
        view = view[view["cuisine"].isin(pick_cuisine)]
    if pick_level:
        view = view[view["risk_level"].isin(pick_level)]
    if query:
        view = view[view["name"].str.contains(query, case=False, na=False)]

    k = st.columns(5)
    k[0].metric("Monitored", len(view))
    k[1].metric("High risk", int((view["risk_level"] == "high").sum()), help="risk at least 2× the typical restaurant")
    k[2].metric("Medium risk", int((view["risk_level"] == "medium").sum()), help="risk 1.5–2× the typical restaurant")
    k[3].metric("Spiked (3 mo)", int(view["spiking_aspects"].map(len).gt(0).sum()),
                help="restaurants with an M3 complaint spike this month or in the two before "
                     "(the Alerts page lists only this month's spikes)")
    k[4].metric("Avg rating (3 mo)", f"{view['rating_3m'].mean():.2f} ★" if len(view) else "–")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        table = view.assign(
            level=view["risk_level"].map(lambda lv: STATUS[lv][1]),
            spikes=view["spiking_aspects"].map(lambda s: ", ".join(s) if len(s) else "—"),
        )[["name", "cuisine", "level", "risk", "risk_lift", "rating_3m", "rating_6m", "reviews_3m", "top_aspect", "spikes",
            "business_id"]]
        with card("ranking", "Restaurants by decline risk",
                  "Calibrated chance of a ≥0.3★ drop over the next 3 months. Select a row to open the restaurant."):
            event = st.dataframe(
                table, hide_index=True, use_container_width=True, height=470, on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "name": "Restaurant", "cuisine": "Cuisine", "level": "Level",
                    "risk": st.column_config.NumberColumn("Decline risk", format="percent"),
                    "risk_lift": st.column_config.NumberColumn("× typical", format="%.1f×"),
                    "rating_3m": st.column_config.NumberColumn("Rating 3 mo", format="%.2f"),
                    "rating_6m": st.column_config.NumberColumn("Rating 6 mo", format="%.2f"),
                    "reviews_3m": st.column_config.NumberColumn("Reviews 3 mo"),
                    "top_aspect": "Main driver", "spikes": "Complaint spikes", "business_id": None})
        if event.selection.rows:
            st.session_state["bid"] = table.iloc[event.selection.rows[0]]["business_id"]
            st.switch_page(PAGES["restaurant"])
    with right:
        hist = alt.Chart(view).mark_bar(color=ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("risk:Q", bin=alt.Bin(maxbins=20), title="Decline risk", axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", title="Restaurants"),
            tooltip=[alt.Tooltip("risk:Q", bin=alt.Bin(maxbins=20), title="Risk band", format=".0%"),
                     alt.Tooltip("count():Q", title="Restaurants")]).properties(height=170)
        with card("hist", "Risk distribution", "Most restaurants sit near the typical level",
                  data=view[["name", "risk"]].sort_values("risk", ascending=False)):
            chart(hist)
        drivers = view["top_aspect"].value_counts().rename_axis("aspect").reset_index(name="restaurants")
        bars = alt.Chart(drivers).mark_bar(cornerRadiusEnd=3, height=14).encode(
            y=alt.Y("aspect:N", sort="-x", title=None),
            x=alt.X("restaurants:Q", title="Restaurants"),
            color=alt.Color("aspect:N", scale=ASPECT_SCALE, legend=None),
            tooltip=["aspect:N", "restaurants:Q"]).properties(height=165)
        with card("drivers", "Main risk driver", "Top contributor to each restaurant's risk", data=drivers):
            chart(bars + bars.mark_text(align="left", dx=4, color=MUTED).encode(text="restaurants:Q"))


# ================================================================== Restaurant
def _verdict(info, m, fc, plan: dict | None = None) -> str:
    """One plain-English sentence answering 'am I in trouble, and why?'.

    Names the same aspect the agent ranks first, so the verdict and the plan never disagree."""
    if plan and plan.get("items"):
        top = plan["items"][0]["aspect"]
    else:
        # counterfactual risk drop only: adding SHAP (logit units) to it would mix units
        top = max(ASPECTS, key=lambda a: fc.get("counterfactual_risk_drop", {}).get(a, 0))
    am = m["aspects"][top]
    market = (f", against a market median of {pct(am['market_median_rate'])}"
              if am.get("market_median_rate") is not None else "")
    return (f"<b>{html.escape(info['name'])}</b> has a <b>{pct(fc['risk'])}</b> chance its rating falls by 0.3★ or "
            f"more over the next 3 months, <b>{fc.get('risk_lift', 1):.1f}×</b> the typical restaurant "
            f"({pct(fc.get('base_rate'))}). Fix <b>{top}</b> first: {pct(am['complaint_rate_3m'])} of recent reviews "
            f"complain about it{market}.")


PLANNER_BADGE = {"claude": "Claude agent", "open_model": "Open-model agent"}


def _plan_section(bid: str):
    """The verified action plan: summary, one card per ranked aspect with its evidence, the verifier log and
    the usefulness rating form."""
    with st.spinner("Running the agent…"):
        plan = load("plan", bid)
    v = plan.get("verification", {})
    planner = plan.get("planner", "rule_based")
    head = st.columns([5, 1], vertical_alignment="bottom")
    head[0].markdown(f"<div class='rv-title' style='font-size:1.25rem;margin-top:12px'>What to do first</div>"
                     f"<p class='rv-sub' style='margin-bottom:10px'>{PLANNER_BADGE.get(planner, 'Rule-based planner')} · "
                     f"{v.get('claims_verified', 0)}/{v.get('claims_total', 0)} claims verified against your reviews · "
                     f"{len(plan.get('tool_calls', []))} tool calls · generated {plan.get('generated_at', '')}</p>",
                     unsafe_allow_html=True)
    if head[1].button("Re-run", icon=":material/refresh:", use_container_width=True,
                      help="Re-run the agent (rule-based unless the server sets RADVIEW_ALLOW_LLM_REFRESH=1, "
                           "because LLM runs cost money)"):
        load.clear()
        client().plan(bid, refresh=True)
        st.rerun()
    with card("plan-summary"):
        st.markdown(md(html.escape(plan["summary"])))
    if not plan["items"]:
        st.success("No pressing issues found — nothing to fix right now.")
    for it in plan["items"]:
        badges = []
        if it.get("star_cost"):
            badges.append(f"≈{it['star_cost']:.2f}★ cost vs market")
        if (it.get("relative_risk_drop") or 0) >= 0.01:
            badges.append(f"fixing cuts risk by {it['relative_risk_drop']:.0%}")
        badges.append(EFFORT_LABEL.get(it.get("effort"), ""))
        badge_html = " ".join(f"<span class='rv-tag'>{b}</span>" for b in badges if b)
        with card(f"plan-item-{it['rank']}"):
            st.markdown(f"<span class='rv-rank' style='background:{ASPECT_COLORS.get(it['aspect'], INK)}'>"
                        f"{it['rank']}</span><b>{it['aspect'].capitalize()}</b>"
                        f"<span class='rv-meta'>{html.escape(it['headline'])}</span> {badge_html}",
                        unsafe_allow_html=True)  # headline may be LLM text
            if it["actions"]:
                st.markdown(f"**Do this:** {md(it['actions'][0])}")
            ev, act = st.columns([3, 2], gap="large")
            with ev:
                for c in it["claims"]:
                    if c.get("type") == "review":
                        st.markdown(f"> {md(c['text'])}  \n> <span class='rv-ok'>VERIFIED</span>"
                                    f"<span class='rv-meta'>review {html.escape(', '.join(c.get('review_ids', [])))}"
                                    f"</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"{md(c['text'])} <span class='rv-ok'>VERIFIED</span>", unsafe_allow_html=True)
            with act:
                if len(it["actions"]) > 1:
                    st.markdown("**Also consider**")
                    for a in it["actions"][1:]:
                        st.markdown(f"- {md(a)}")
                st.caption(it["why"])
    rejected = plan.get("rejected_claims") or []
    with st.expander(f"Verifier log · {len(rejected)} claim(s) rejected before you saw them"):
        if rejected:
            st.dataframe(pd.DataFrame(rejected), hide_index=True, use_container_width=True)
        else:
            st.write("Every drafted claim passed the citation, quotation, relevance and statistic checks. "
                     "They check that each claim is traceable to a review or a returned figure, not that it is true.")
    with st.expander("Rate this action plan"):
        with st.form(f"fb-{bid}", clear_on_submit=True):
            score = st.slider("How useful is this plan?", 1, 5, 4)
            comment = st.text_input("Comment (optional)")
            if st.form_submit_button("Submit rating"):
                summary = client().feedback(bid, score, comment)
                st.success(f"Thanks! Average usefulness so far: {summary['mean_usefulness']} / 5 ({summary['n']} ratings)")


def page_restaurant():
    """One restaurant, top to bottom in the order an owner asks: am I in trouble, is my rating moving, what do
    people complain about, is it getting worse, what do I do first, then the deeper explanations."""
    rows = pd.DataFrame(load("restaurants"))
    ids = rows["business_id"].tolist()
    names = dict(zip(rows["business_id"],
                     rows["name"] + " — " + (rows["risk"] * 100).round().astype(int).astype(str) + "% risk"))
    title_col, pick_col = st.columns([3, 2], vertical_alignment="bottom")
    default = st.session_state.get("bid", ids[0])
    bid = pick_col.selectbox("Restaurant", ids, index=ids.index(default) if default in ids else 0,
                             format_func=names.get)
    if bid != st.session_state.get("bid"):
        st.session_state.pop("issue", None)
    st.session_state["bid"] = bid
    d = load("restaurant", bid)
    info, m, fc, peers = d["info"], d["metrics"], d["forecast"], d["peers"]
    tl = pd.DataFrame(load("timeline", bid))
    tl["date"] = pd.to_datetime(tl["month"])
    with title_col:
        header(html.escape(info["name"]),
               html.escape(f"{info['cuisine']} · {info['address']}, {info['city']} · analysis for {m['month']}"),
               eyebrow="Restaurant")

    # --- Am I in trouble? -------------------------------------------------------
    with st.spinner("Running the agent…"):
        plan = load("plan", bid)
    with card("verdict"):
        st.markdown(f"{status_badge(fc['risk_level'])}<div class='rv-verdict'>{_verdict(info, m, fc, plan)}</div>",
                    unsafe_allow_html=True)
    spikes = [a for a, v in m["aspects"].items() if v["spike"]]
    k = st.columns(4)
    k[0].metric("Chance rating falls", pct(fc["risk"]), delta=f"{fc.get('risk_lift', 1):.1f}× typical", delta_color="off",
                help="next 3 months, ≥0.3★, calibrated probability; typical restaurant = mean predicted risk of "
                     "the restaurants active this month")
    delta = None if m["rating_3m"] is None or m["rating_6m"] is None else round(m["rating_3m"] - m["rating_6m"], 2)
    k[1].metric("Average rating (3 mo)", f"{m['rating_3m']:.2f} ★" if m["rating_3m"] else "–",
                delta=f"{delta:+.2f} vs 6 mo" if delta is not None else None)
    k[2].metric("Reviews (3 mo)", m["reviews_3m"])
    k[3].metric("Spikes (3 mo)", len(spikes), help=", ".join(spikes) if spikes else "nothing spiked recently")

    # --- Is my rating moving? / What are people complaining about? -------------
    left, right = st.columns(2, gap="medium")
    with left:
        shown = tl.tail(36)
        hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
        base = alt.Chart(shown).encode(x=alt.X("date:T", title=None))
        rating = alt.layer(
            base.mark_line(strokeWidth=2, color=ACCENT, point=alt.OverlayMarkDef(size=22, color=ACCENT)).encode(
                y=alt.Y("mean_stars:Q", title="Stars", scale=alt.Scale(domain=[1, 5]))),
            base.mark_rule(color=MUTED).encode(
                opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
                tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("mean_stars:Q", title="Mean", format=".2f"),
                         alt.Tooltip("n_reviews:Q", title="Reviews")]).add_params(hover),
        ).properties(height=250)
        with card("rating", "Average rating by month", "Gaps are months with no reviews",
                  data=shown[["month", "mean_stars", "n_reviews"]]):
            chart(rating)
    with right:
        comp = pd.DataFrame([{"aspect": a, "rate": v["complaint_rate_3m"], "market": v.get("market_median_rate"),
                              "z": v["robust_z"]} for a, v in m["aspects"].items()])
        bars = alt.Chart(comp).mark_bar(cornerRadiusEnd=3, height=18).encode(
            y=alt.Y("aspect:N", sort=ASPECTS, title=None, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("rate:Q", title="Share of recent reviews complaining", axis=alt.Axis(format="%")),
            color=alt.Color("aspect:N", scale=ASPECT_SCALE, legend=None),
            tooltip=[alt.Tooltip("aspect:N", title="Issue"), alt.Tooltip("rate:Q", title="You", format=".1%"),
                     alt.Tooltip("market:Q", title="Market median", format=".1%"),
                     alt.Tooltip("z:Q", title="Robust z", format=".1f")])
        ticks = alt.Chart(comp).mark_tick(color=INK, thickness=2, size=24).encode(
            y=alt.Y("aspect:N", sort=ASPECTS), x="market:Q")
        with card("complaints", "What people complain about", "Your rate (bar) against the market median (line)",
                  data=comp):
            chart(alt.layer(bars, ticks).properties(height=200))
            worst = max(ASPECTS, key=lambda a: m["aspects"][a]["complaint_rate_3m"])
            current = st.session_state.get("issue") if st.session_state.get("issue") in ASPECTS else worst
            chosen = st.pills("Show the trend for", ASPECTS, default=current, key=f"issue-pills-{bid}",
                              label_visibility="collapsed")
            st.session_state["issue"] = chosen or current

    # --- Is it getting worse? ---------------------------------------------------------
    issue = st.session_state["issue"]
    sub = tl.tail(36)
    series = ["this restaurant", "market median"]
    long = pd.concat([
        pd.DataFrame({"date": sub["date"], "month": sub["month"], "series": series[0], "value": sub[f"{issue}_neg_rate_3m"]}),
        pd.DataFrame({"date": sub["date"], "month": sub["month"], "series": series[1], "value": sub[f"{issue}_market_rate"]}),
    ])
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("value:Q", title="Share of reviews", axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(domain=series, range=[ASPECT_COLORS[issue], COMPARE]), title=None,
                        legend=alt.Legend(orient="top-left", symbolStrokeWidth=3)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=series, range=[[1, 0], [4, 4]]), legend=None),
        tooltip=["month:N", "series:N", alt.Tooltip("value:Q", format=".1%")])
    dots = alt.Chart(sub[sub[f"{issue}_spike"] == 1]).mark_point(
        filled=True, size=70, color=RISK, stroke=CARD, strokeWidth=2).encode(
        x="date:T", y=f"{issue}_neg_rate_3m:Q",
        tooltip=[alt.Tooltip("month:N", title="Spike"), alt.Tooltip(f"{issue}_z:Q", title="Robust z", format=".1f")])
    with card("issue", f"{issue.capitalize()} complaints over time",
              "Share of the last 3 months' reviews, yours against the market median; red points are complaint spikes",
              data=sub[["month", f"{issue}_rate", f"{issue}_neg_rate_3m", f"{issue}_market_rate", f"{issue}_z",
                        f"{issue}_spike"]]):
        chart(alt.layer(lines, dots).properties(height=210))

    # --- What do I do on Monday? --------------------------------------------------------
    _plan_section(bid)

    # --- Deeper dives ---------------------------------------------------------------
    st.markdown("<div class='rv-title' style='font-size:1.25rem;margin-top:18px'>Details</div>", unsafe_allow_html=True)
    with card("details"):
        tab_drv, tab_risk, tab_peer, tab_rev = st.tabs(["Why at risk", "Risk history", "Peers", "Reviews"])
        with tab_drv:
            st.caption("Two independent explanations of the same forecast. SHAP splits the score across inputs; the "
                       "counterfactual asks how much the risk drops if one aspect looked like the market norm.")
            c1, c2 = st.columns(2, gap="large")
            attr = pd.DataFrame([{"driver": k_, "value": v} for k_, v in fc["attributions"].items()])
            attr["direction"] = attr["value"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
            with c1:
                st.markdown("<div class='rv-card-title'>SHAP contribution</div>", unsafe_allow_html=True)
                chart(alt.Chart(attr).mark_bar(cornerRadiusEnd=3, height=12).encode(
                    y=alt.Y("driver:N", sort=alt.EncodingSortField("value", order="descending"), title=None,
                            axis=alt.Axis(labelOverlap=False, labelLimit=160)),
                    x=alt.X("value:Q", title="Contribution (logit)"),
                    color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                                   range=[INCREASES, DECREASES]), title=None),
                    tooltip=["driver:N", alt.Tooltip("value:Q", format="+.3f")]).properties(height=alt.Step(26)))
            cf = pd.DataFrame([{"aspect": a, "risk drop if fixed": fc.get("counterfactual_risk_drop", {}).get(a, 0),
                                "star cost": fc.get("star_cost", {}).get(a, 0)} for a in ASPECTS])
            with c2:
                st.markdown("<div class='rv-card-title'>Risk reduction at the market norm</div>", unsafe_allow_html=True)
                chart(alt.Chart(cf).mark_bar(cornerRadiusEnd=3, height=14).encode(
                    y=alt.Y("aspect:N", sort="-x", title=None),
                    color=alt.Color("aspect:N", scale=ASPECT_SCALE, legend=None),
                    x=alt.X("risk drop if fixed:Q", title="Counterfactual risk drop", axis=alt.Axis(format="%")),
                    tooltip=["aspect:N", alt.Tooltip("risk drop if fixed:Q", format=".1%"),
                             alt.Tooltip("star cost:Q", title="Star cost", format=".2f")]).properties(height=170))
        with tab_risk:
            split = ((load("evaluation").get("m4") or {}).get("split") or {})
            test_start = (split.get("test") or "").split("..")[0]
            marks = []
            if test_start:  # before the test period the model has seen the outcomes: those points are in-sample
                marks.append(alt.Chart(pd.DataFrame({"date": [pd.to_datetime(test_start)]})).mark_rule(
                    color=MUTED, strokeDash=[2, 2]).encode(x="date:T"))
            risk = alt.layer(
                alt.Chart(tl).mark_area(color=ACCENT, opacity=0.08).encode(
                    x=alt.X("date:T", title=None), y=alt.Y("risk:Q", title="Risk", axis=alt.Axis(format="%"))),
                alt.Chart(tl).mark_line(strokeWidth=2, color=ACCENT).encode(
                    x="date:T", y="risk:Q", tooltip=["month:N", alt.Tooltip("risk:Q", format=".0%")]),
                alt.Chart(pd.DataFrame({"y": [1.5 * fc["base_rate"], 2.0 * fc["base_rate"]]})).mark_rule(
                    strokeDash=[4, 4], color=COMPARE).encode(y="y:Q"),
                *marks,
            ).properties(height=220)
            st.caption("Predicted decline risk each month; dashed lines mark the medium (1.5×) / high (2×) "
                       "thresholds relative to the typical restaurant."
                       + (f" Months before {test_start} (dotted line) were used to train or calibrate the model, so "
                          "the risk there is in-sample and looks more accurate than a real forecast." if test_start else ""))
            chart(risk)
        with tab_peer:
            st.caption(f"Complaint rate against {peers['basis']} ({peers['n_peers']} with recent reviews)")
            pdf = pd.concat([pd.DataFrame({"aspect": a, "who": ["This restaurant", "Peer median"],
                                           "rate": [v["mine"], v["peer_median"]], "percentile": v["percentile"]})
                             for a, v in peers["aspects"].items()])
            st.altair_chart(alt.Chart(pdf).mark_bar(cornerRadiusEnd=3, height=11).encode(
                y=alt.Y("who:N", title=None, axis=None),
                x=alt.X("rate:Q", title="Share of reviews complaining", axis=alt.Axis(format="%")),
                color=alt.Color("who:N", scale=alt.Scale(domain=["This restaurant", "Peer median"],
                                                         range=[ACCENT, COMPARE]), title=None),
                row=alt.Row("aspect:N", sort=ASPECTS, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                tooltip=["aspect:N", "who:N", alt.Tooltip("rate:Q", format=".1%"),
                         alt.Tooltip("percentile:Q", title="Percentile among peers", format=".0f")],
            ).properties(height=32, width=520))
        with tab_rev:
            c1, c2 = st.columns(2)
            aspect = c1.selectbox("Aspect", ["(all)"] + ASPECTS)
            sentiment = c2.selectbox("Sentiment", ["negative", "positive"], disabled=aspect == "(all)")
            for r in load("reviews", bid, None if aspect == "(all)" else aspect, sentiment, 30):
                tags = " ".join(f"<span class='rv-tag'>{a} {'+' if v == 1 else '−'}</span>"
                                for a, v in r["aspects"].items() if v)
                st.markdown(f"<b>{'★' * int(r['stars'])}{'☆' * (5 - int(r['stars']))}</b>"
                            f"<span class='rv-meta'>{r['date']}</span> {tags}", unsafe_allow_html=True)
                st.markdown(md(html.escape(r["text"])))
                st.divider()


# ================================================================== Alerts
def page_alerts():
    """This month's complaint spikes across all restaurants, filterable by severity; a row opens the restaurant."""
    header("Alerts", "Aspects whose complaint rate broke away from the restaurant's own 12-month baseline this month "
                     "(M3 robust z-score). Select a row to open the restaurant.")
    rows = pd.DataFrame(load("alerts"))
    if rows.empty:
        st.success("Nothing is spiking right now.")
        return
    sev = st.segmented_control("Severity", list(SEVERITY), selection_mode="multi", default=list(SEVERITY),
                               format_func=SEVERITY.get, label_visibility="collapsed")
    rows = rows[rows["severity"].isin(sev or list(SEVERITY))]
    table = rows.assign(level=rows["severity"].map(SEVERITY))[
        ["name", "aspect", "level", "robust_z", "complaint_rate_3m", "market_median_rate", "risk", "business_id"]]
    with card("alerts", f"{len(table)} spikes this month",
              "The Portfolio also counts spikes from the two months before, so its number is larger"):
        event = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun",
                             selection_mode="single-row",
                             column_config={
                                 "name": "Restaurant", "aspect": "Issue", "level": "Severity",
                                 "robust_z": st.column_config.NumberColumn("Robust z", format="%.1f"),
                                 "complaint_rate_3m": st.column_config.NumberColumn("Complaint rate", format="percent"),
                                 "market_median_rate": st.column_config.NumberColumn("Market median", format="percent"),
                                 "risk": st.column_config.NumberColumn("Decline risk", format="percent"),
                                 "business_id": None})
    if event.selection.rows:
        st.session_state["bid"] = table.iloc[event.selection.rows[0]]["business_id"]
        st.session_state["issue"] = table.iloc[event.selection.rows[0]]["aspect"]
        st.switch_page(PAGES["restaurant"])


# ================================================================== Try a review
def page_analyse():
    """Live M2 demo: label pasted reviews and highlight the sentence behind each complaint."""
    header("Try a review", "Paste one review per line. The production aspect model (M2) labels each aspect and "
                           "highlights the sentence that carries each complaint, the same check the agent's verifier uses.")
    with card("analyse-input"):
        # one clear issue per review: the model reads whole reviews, and in mixed ones it can drag a praised aspect
        # negative too, so the live example sticks to cases it handles reliably
        text = st.text_area("Reviews", height=150, label_visibility="collapsed", value=(
            "We waited 40 minutes to be seated and the host was rude about it.\n"
            "The restroom was filthy and smelled bad. We left before ordering.\n"
            "Way too expensive for what you get.\n"
            "Lovely patio with a cozy atmosphere. Our server was friendly and attentive all night."))
        go = st.button("Analyse", type="primary", icon=":material/arrow_forward:")
    if go:
        texts = [t.strip() for t in text.splitlines() if t.strip()][:50]
        for i, res in enumerate(client().analyse(texts)):
            with card(f"analyse-{i}"):
                body = html.escape(res["text"])  # pasted text is shown as text, never as HTML
                for a, sent in res["complaint_sentences"].items():
                    sent = html.escape(sent)
                    body = body.replace(sent, f"<mark style='background:{ASPECT_COLORS[a]}2e;color:inherit;"
                                              f"border-bottom:2px solid {ASPECT_COLORS[a]};border-radius:3px;"
                                              f"padding:0 2px'>{sent}</mark>")
                st.markdown(body, unsafe_allow_html=True)
                chips = [f"<span class='rv-badge'><span class='rv-dot' style='--c:"
                         f"{STATUS['high' if lab == 'negative' else 'low'][0]}'></span>"
                         f"{a} · {lab} · {res['confidence'][a]:.0%}</span>"
                         for a, lab in res["aspects"].items() if lab != "not mentioned"]
                st.markdown(" ".join(chips) or f"<span style='color:{MUTED}'>no aspects detected</span>",
                            unsafe_allow_html=True)


# ================================================================== About
def page_about():
    """Architecture diagram and a one-line summary of each module."""
    header("About Radview", "Radview warns restaurant owners before their rating falls and tells them, with review "
                            "evidence, which problem to fix first. Six modules, deployed as Docker containers on Kubernetes.")
    node_fill, node_ink = (CARD, INK)
    with card("about-arch", "Architecture"):
        st.graphviz_chart(f"""
digraph {{ rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="{GRID}", fillcolor="{node_fill}",
        fontcolor="{node_ink}", penwidth=1.2];
  edge [color="{MUTED}", fontcolor="{MUTED}", fontsize=9, arrowsize=0.7];
  data [label="Yelp Open Dataset\\nSemEval-2014 ABSA"];
  m1 [label="M1 Ingestion\\nstream · clean · dedupe"];
  m2 [label="M2 Aspect classifier\\nDistilBERT (PyTorch)"];
  m3 [label="M3 Trend detector\\nEWMA + robust z"];
  m4 [label="M4 Decline forecaster\\nGRU + SHAP + counterfactual"];
  m5 [label="M5 Agent (LangGraph)\\ntools → verify", color="{ACCENT}"];
  m6 [label="M6 FastAPI + Streamlit"];
  cron [label="Weekly CronJob"];
  data -> m1 -> m2 -> m3 -> m4 -> m5 -> m6; m2 -> m4; m2 -> m5 [label="evidence"]; cron -> m1; }}
""")
    with card("about-modules", "Modules"):
        st.markdown("""
| Module | What it does |
|---|---|
| **M1** | Streams the 5 GB Yelp review file, keeps one city's restaurants, cleans & de-duplicates, writes parquet. |
| **M2** | Fine-tuned DistilBERT tags each review's sentiment toward food, service, cleanliness, price and ambience, reading the text. Trained by weak supervision (about 4 in 10 training labels still come from the star rating); chosen on a SemEval-2014 dev set and evaluated on the human-labelled SemEval-2014 test set. |
| **M3** | Flags aspects whose complaint rate jumps above the restaurant's own history (robust z-score). |
| **M4** | Predicts a significant ≥0.3★ drop over the next 3 months; a PyTorch GRU reads 12 months of signals incl. market-relative features. Explained twice — SHAP and a counterfactual — plus a star-cost estimate. |
| **M5** | A LangGraph agent: an LLM calls the analytics tools itself and submits a structured plan; a mechanical verifier rejects claims that fail its citation, quotation, relevance or statistic checks (traceability, not truth). A rule-based planner is the fallback and the baseline. |
| **M6** | REST API (`/docs`) + this dashboard; one Docker image; Kubernetes Job, weekly CronJob and Deployments. |
""")
    st.caption(f"Data client mode: {client().mode} · theme: {'dark' if IS_DARK else 'light'}")


PAGES = {
    "portfolio": st.Page(page_portfolio, title="Portfolio", icon=":material/space_dashboard:", url_path="portfolio",
                         default=True),
    "restaurant": st.Page(page_restaurant, title="Restaurant", icon=":material/storefront:", url_path="restaurant"),
    "alerts": st.Page(page_alerts, title="Alerts", icon=":material/notifications:", url_path="alerts"),
    "analyse": st.Page(page_analyse, title="Try a review", icon=":material/rate_review:", url_path="analyse"),
    "about": st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
}
nav = st.navigation(list(PAGES.values()))
with st.sidebar:
    _meta = load("meta")
    st.markdown(f"<div class='rv-side-foot'>Data <b>{html.escape(str(_meta['source']))}</b> · through "
                f"<b>{_meta['latest_month']}</b><br>{_meta['restaurants']} restaurants · {_meta['reviews']:,} reviews"
                f"</div>", unsafe_allow_html=True)
nav.run()
