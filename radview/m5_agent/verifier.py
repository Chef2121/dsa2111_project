"""M5 - claim verifier.

Every claim in an action plan must be traceable. A claim passes only if:

* review evidence - each cited review exists, belongs to this restaurant,
  is classified by M2 as *negative* on the claimed aspect, the quoted text
  appears verbatim in that review, and the quote *on its own* is classified
  by M2 as a complaint about the aspect (so a positive sentence lifted from a
  negative review cannot be passed off as a complaint);
* metric evidence - every percentage written in the claim matches (within
  2 percentage points) one of the metrics the tools returned for that aspect.

Claims that fail are removed from the final plan and counted against the
groundedness score (share of drafted claims that pass).
"""
from __future__ import annotations

import re

from radview.config import LABEL_NEG
from radview.m5_agent.tools import STORE, Store, is_complaint

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip('"“”').lower()


def verify_claim(claim: dict, business_id: str, aspect: str, metrics: dict, store: Store = STORE) -> dict:
    reasons = []
    review_ids = claim.get("review_ids") or []
    quote = claim.get("quote")
    if claim.get("type", "review") == "review":
        if not review_ids:
            reasons.append("no review cited")
        reviews = store.reviews
        quote_found = quote is None
        for rid in review_ids:
            if rid not in reviews.index:
                reasons.append(f"review {rid} does not exist")
                continue
            r = reviews.loc[rid]
            if r["business_id"] != business_id:
                reasons.append(f"review {rid} belongs to another restaurant")
            if int(r[f"{aspect}_label"]) != LABEL_NEG:
                reasons.append(f"review {rid} is not a {aspect} complaint")
            if quote and _norm(quote) in _norm(r["text"]):
                quote_found = True
        if not quote_found:
            reasons.append("quote not found verbatim in cited reviews")
        elif quote and not is_complaint(quote, aspect):
            reasons.append(f"quoted sentence is not a {aspect} complaint")
    # numeric check applies to every claim type
    allowed = [100 * v for v in metrics.values() if isinstance(v, (int, float))]
    own_words = claim.get("text", "").replace(quote or "\0", "")  # figures inside a verbatim quote are the reviewer's
    for m in _PCT.finditer(own_words):
        val = float(m.group(1))
        if not any(abs(val - a) <= 2.0 for a in allowed):
            reasons.append(f"figure {val}% not supported by metrics")
    return {**claim, "verified": not reasons, "reasons": reasons}


def verify_plan(plan: dict, store: Store = STORE) -> dict:
    """Annotate each claim with verified/reasons and compute groundedness."""
    total = passed = 0
    for item in plan.get("items", []):
        aspect = item["aspect"]
        metrics = item.get("_metrics", {})
        checked = [verify_claim(c, plan["business_id"], aspect, metrics, store) for c in item.get("claims", [])]
        item["claims"] = checked
        total += len(checked)
        passed += sum(c["verified"] for c in checked)
    plan["verification"] = {"claims_total": total, "claims_verified": passed,
                            "groundedness": round(passed / total, 4) if total else None}
    return plan
