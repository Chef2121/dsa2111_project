"""M5 - claim verifier.

Every claim in an action plan must be traceable to what the analytics produced.
The verifier is deliberately mechanical - not another LLM call - because a check
that can itself hallucinate is not independent evidence. Each claim is tested on
five axes:

    citation   every cited review exists, belongs to this restaurant and was
               actually returned to the agent by ``get_evidence_reviews``
    quotation  the quote appears verbatim in one of the cited reviews
    relevance  M2 classifies the cited review as negative on the claimed aspect
               *and* classifies the quoted sentence on its own as a complaint
               about it (so a positive sentence lifted from a negative review
               cannot be passed off as a complaint)
    statistic  every percentage matches a complaint/market rate the tools
               returned (+-2 points) and every star figure matches a rating or
               star-cost value (+-0.05)
    substance  a metric claim must state at least one checkable figure

Claims that fail are removed from the final plan but counted: the share that
passes is the plan's groundedness, M5's headline metric, and the per-axis
failure counts show *how* a planner goes wrong.
"""
from __future__ import annotations

import re

from radview.config import LABEL_NEG
from radview.m5_agent.tools import STORE, Store, is_complaint

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_STARS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:★|stars?\b)")
PCT_TOL, STAR_TOL = 2.0, 0.05
AXES = ("citation", "quotation", "relevance", "statistic", "substance")


def _norm(s: str) -> str:
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
    rates = [100 * v for v in metrics.get("rates", {}).values() if isinstance(v, (int, float))]
    stars = [v for v in metrics.get("stars", {}).values() if isinstance(v, (int, float))]
    pcts, star_figs = _PCT.findall(own_words), _STARS.findall(own_words)
    for m in pcts:
        if not any(abs(float(m) - a) <= PCT_TOL for a in rates):
            fails.append(("statistic", f"figure {m}% not supported by the metrics"))
    for m in star_figs:
        if not any(abs(float(m) - a) <= STAR_TOL for a in stars):
            fails.append(("statistic", f"figure {m} stars not supported by the metrics"))
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
