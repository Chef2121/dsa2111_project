"""M5 - claim verifier.

Every claim in an action plan must be traceable to what the analytics produced.
The verifier is deliberately mechanical - not another LLM call - because a check
that can itself hallucinate is not independent evidence. Each claim is tested on
five axes:

    citation   every cited review exists, belongs to this restaurant and was
               actually returned to the agent by ``get_evidence_reviews``
    quotation  a review claim carries a non-empty quote, and the quote appears
               verbatim in one of the cited reviews
    relevance  M2 classifies the cited review as negative on the claimed aspect
               *and* classifies the quoted sentence on its own as a complaint
               about it (so a positive sentence lifted from a negative review
               cannot be passed off as a complaint)
    statistic  every percentage / star figure matches (+-2 points / +-0.05 stars)
               the metric its own clause refers to: the market median only in a
               clause that mentions the market, the decline risk only next to
               "risk" or "chance", a star cost only next to "cost" ... - so a real
               number attached to the wrong quantity fails. Multipliers such as
               "doubled" or "3x" cannot be checked and fail too.
    substance  a metric claim must state at least one checkable figure

Claims that fail are removed from the final plan but counted: the share that
passes is the plan's groundedness, M5's headline metric, and the per-axis
failure counts show *how* a planner goes wrong.

Limits (stated in the report): the checks are about traceability, not truth;
relevance relies on the same M2 model that selected the evidence; and figures
written as words ("one in four") are not recognised.
"""
from __future__ import annotations

import re

from radview.config import LABEL_NEG
from radview.m5_agent.tools import STORE, Store, is_complaint

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_STARS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:★|stars?\b)")
_MULTIPLIER = re.compile(r"\b(?:doubled|tripled|quadrupled|halved|twice as|\d+(?:\.\d+)?\s*(?:x|×|times)\b)",
                         re.IGNORECASE)
_CLAUSE = re.compile(r"[;,]|\bversus\b|\bvs\.?\b|\bcompared (?:with|to)\b|\bwhile\b|\bwhereas\b|\bbut\b",
                     re.IGNORECASE)
PCT_TOL, STAR_TOL, LIFT_TOL = 2.0, 0.05, 0.1
AXES = ("citation", "quotation", "relevance", "statistic", "substance")

# Which words a clause must contain before a figure may be matched against each metric. ``None`` = no cue
# needed (the restaurant's own current complaint rate). Metrics not listed here need no cue either.
RATE_CUES: dict[str, tuple[str, ...] | None] = {
    "complaint_rate_3m": None,
    "complaint_rate_prev6m": ("before", "previous", "prior", "earlier", "six months", "6 months", "used to",
                              "up from", "down from", "rose from", "fell from"),
    "praise_rate_3m": ("praise", "positive", "compliment", "happy", "liked", "loved"),
    "market_median_rate": ("market", "typical", "median", "city", "average restaurant"),
    "peer_median": ("peer", "similar", "comparable", "competitor", "other restaurants"),
    "risk": ("risk", "chance", "probability", "likel"),
    "risk_drop_if_fixed": ("risk", "chance", "probability", "reduc", "lower", "cut", "drop"),
}
STAR_CUES: dict[str, tuple[str, ...] | None] = {
    "star_cost": ("cost", "lose", "losing", "lost", "drag"),
    "rating_3m": ("rating", "rated", "average", "currently", "recent", "3 months", "three months"),
    "rating_6m": ("rating", "rated", "average", "six months", "6 months"),
}


def _supported(value: float, metrics: dict, cues: dict, clause: str, tol: float, scale: float = 1.0) -> bool:
    """True if ``value`` matches, within ``tol``, a metric whose cue words appear in ``clause`` (metrics with
    no cue list always qualify).
    """
    low = clause.lower()
    for name, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        need = cues.get(name)
        if need is not None and not any(w in low for w in need):
            continue
        if abs(value - scale * v) <= tol:
            return True
    return False


def figure_failures(text: str, metrics: dict) -> list[str]:
    """Statistic failures for the figures in ``text`` (clause by clause, see the module docstring)."""
    fails = []
    for clause in _CLAUSE.split(text):
        for m in _PCT.findall(clause):
            if not _supported(float(m), metrics.get("rates", {}), RATE_CUES, clause, PCT_TOL, 100.0):
                fails.append(f"figure {m}% not supported by the metric this sentence refers to")
        for m in _STARS.findall(clause):
            if not _supported(float(m), metrics.get("stars", {}), STAR_CUES, clause, STAR_TOL):
                fails.append(f"figure {m} stars not supported by the metric this sentence refers to")
    for m in _MULTIPLIER.findall(text):
        fails.append(f"'{m}' is a multiplier no tool returned")
    return fails


def unsupported_text_figures(text: str, metrics_list: list[dict], risk_lift: float | None = None) -> list[str]:
    """Figures in free text (summary, headline) that match *no* figure the tools returned for this plan.

    Free text mixes aspects, so this is looser than ``figure_failures``: any returned value of the right
    kind is accepted, and a multiplier must be the restaurant's risk lift.
    """
    rates = [100 * v for m in metrics_list for v in m.get("rates", {}).values() if isinstance(v, (int, float))]
    stars = [v for m in metrics_list for v in m.get("stars", {}).values() if isinstance(v, (int, float))]
    bad = [f"{p}%" for p in _PCT.findall(text) if not any(abs(float(p) - r) <= PCT_TOL for r in rates)]
    bad += [f"{s} stars" for s in _STARS.findall(text) if not any(abs(float(s) - v) <= STAR_TOL for v in stars)]
    for m in _MULTIPLIER.findall(text):
        num = re.match(r"\d+(?:\.\d+)?", m)
        if not (num and risk_lift is not None and abs(float(num.group()) - risk_lift) <= LIFT_TOL):
            bad.append(m)
    return bad


def _norm(s: str) -> str:
    """Whitespace-collapsed, lower-case text without surrounding quotes, for verbatim-quote matching."""
    return re.sub(r"\s+", " ", s).strip().strip('"“”').lower()


def verify_claim(claim: dict, business_id: str, aspect: str, metrics: dict, store: Store = STORE,
                 retrieved_ids: set[str] | None = None) -> dict:
    """Return the claim annotated with ``verified`` and a list of ``(axis, reason)`` failures.

    ``metrics`` = {"rates": {name: fraction}, "stars": {name: value}} for the claim's aspect.
    ``retrieved_ids`` = review ids the evidence tool returned in this run (None = don't check).
    """
    fails: list[tuple[str, str]] = []
    review_ids = claim.get("review_ids") or []
    quote = claim.get("quote")
    kind = claim.get("type", "review")

    if kind == "review":
        if not review_ids:
            fails.append(("citation", "no review cited"))
        if not (quote or "").strip():
            # without a quote, any text could ride on a real citation
            fails.append(("quotation", "review claim has no quote"))
            quote = None
        quote_found = quote is None
        for rid in review_ids:
            if rid not in store.reviews.index:
                fails.append(("citation", f"review {rid} does not exist"))
                continue
            if retrieved_ids is not None and rid not in retrieved_ids:
                fails.append(("citation", f"review {rid} was never retrieved as evidence"))
            r = store.reviews.loc[rid]
            if r["business_id"] != business_id:
                fails.append(("citation", f"review {rid} belongs to another restaurant"))
            if int(r[f"{aspect}_label"]) != LABEL_NEG:
                fails.append(("relevance", f"review {rid} is not a {aspect} complaint"))
            if quote and _norm(quote) in _norm(r["text"]):
                quote_found = True
        if not quote_found:
            fails.append(("quotation", "quote not found verbatim in cited reviews"))
        elif quote and not is_complaint(quote, aspect):
            fails.append(("relevance", f"quoted sentence is not a {aspect} complaint"))

    # Figures inside a verbatim quote are the reviewer's words, not claims by the agent.
    own_words = claim.get("text", "").replace(quote or "\0", "")
    fails += [("statistic", r) for r in figure_failures(own_words, metrics)]
    pcts, star_figs = _PCT.findall(own_words), _STARS.findall(own_words)
    if kind == "metric" and not pcts and not star_figs:
        fails.append(("substance", "metric claim states no checkable figure"))

    return {**claim, "verified": not fails, "reasons": [f"{a}: {r}" for a, r in fails],
            "failed_axes": sorted({a for a, _ in fails})}


def verify_plan(plan: dict, store: Store = STORE, retrieved_ids: set[str] | None = None) -> dict:
    """Annotate each claim with verified/reasons and compute groundedness + per-axis failures."""
    total = passed = 0
    by_axis = dict.fromkeys(AXES, 0)
    for item in plan.get("items", []):
        aspect = item["aspect"]
        checked = [verify_claim(c, plan["business_id"], aspect, item.get("_metrics", {}), store, retrieved_ids)
                   for c in item.get("claims", [])]
        item["claims"] = checked
        total += len(checked)
        passed += sum(c["verified"] for c in checked)
        for c in checked:
            for a in c["failed_axes"]:
                by_axis[a] += 1
    plan["verification"] = {"claims_total": total, "claims_verified": passed,
                            "groundedness": round(passed / total, 4) if total else None,
                            "failures_by_axis": by_axis}
    return plan
