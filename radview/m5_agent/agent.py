"""M5 - grounded agentic prioritisation (LangGraph).

Workflow (one run per restaurant)::

    gather ──> prioritise ──> retrieve ──> draft ──> verify ──┬──> finalise
                                             ^                │
                                             └── (redraft) ───┘  only in LLM mode, max 2 attempts

* gather      - calls the M2-M4 tools: query_metrics, run_forecast, compare_peers
* prioritise  - ranks aspects by SHAP contribution to decline risk (M4),
                complaint spike z-score (M3) and complaint growth
* retrieve    - get_evidence_reviews for each prioritised aspect
* draft       - an LLM (Claude, if ANTHROPIC_API_KEY is set) or the
                deterministic planner writes a ranked plan citing reviews
* verify      - every claim is checked against the evidence (verifier.py)
* finalise    - unverified claims are dropped; groundedness is reported

The deterministic planner makes the system fully functional offline and is
the reference the LLM output is compared against.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from radview.config import ASPECTS, LLM_MODEL, OUTPUTS_DIR
from radview.m5_agent.playbook import PLAYBOOK, actions_for, detect_sub_themes
from radview.m5_agent.tools import STORE, compare_peers, get_evidence_reviews, query_metrics, run_forecast
from radview.m5_agent.verifier import verify_plan

MAX_ITEMS = 3
MAX_ATTEMPTS = 2
PLANS_DIR = OUTPUTS_DIR / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)


class PlanState(TypedDict, total=False):
    business_id: str
    use_llm: bool
    metrics: dict
    forecast: dict
    peers: dict
    priorities: list[dict]
    evidence: dict[str, list[dict]]
    draft_plan: dict
    attempts: int
    feedback: list[str]
    plan: dict
    mode: str


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def gather(state: PlanState) -> PlanState:
    bid = state["business_id"]
    return {"metrics": query_metrics(bid), "forecast": run_forecast(bid), "peers": compare_peers(bid), "attempts": 0}


def prioritise(state: PlanState) -> PlanState:
    attrs = state["forecast"]["attributions"]
    aspects = state["metrics"]["aspects"]
    pos_shap_total = sum(max(attrs.get(a, 0.0), 0.0) for a in ASPECTS) + 1e-9
    ranked = []
    for a in ASPECTS:
        m = aspects[a]
        shap_part = max(attrs.get(a, 0.0), 0.0) / pos_shap_total
        z_part = min(max(m["robust_z"], 0.0) / 5.0, 1.0)
        growth_part = min(max(m["change"], 0.0) / 0.2, 1.0)
        score = 0.5 * shap_part + 0.3 * z_part + 0.2 * growth_part
        ranked.append({"aspect": a, "priority": round(score, 4), "shap": round(attrs.get(a, 0.0), 4),
                       "robust_z": m["robust_z"], "change": m["change"], "spike": m["spike"],
                       "complaint_rate_3m": m["complaint_rate_3m"]})
    ranked.sort(key=lambda r: -r["priority"])
    chosen = [r for r in ranked if r["priority"] > 0.05 and r["complaint_rate_3m"] > 0][:MAX_ITEMS]
    if not chosen:  # nothing pressing: surface the most-complained-about aspect as a watch item
        chosen = sorted(ranked, key=lambda r: -r["complaint_rate_3m"])[:1]
    return {"priorities": chosen}


def retrieve(state: PlanState) -> PlanState:
    bid = state["business_id"]
    return {"evidence": {p["aspect"]: get_evidence_reviews(bid, p["aspect"]) for p in state["priorities"]}}


def _aspect_metrics(state: PlanState, aspect: str) -> dict:
    m = state["metrics"]["aspects"][aspect]
    return {"complaint_rate_3m": m["complaint_rate_3m"], "complaint_rate_prev6m": m["complaint_rate_prev6m"],
            "praise_rate_3m": m["praise_rate_3m"], "peer_median": state["peers"]["aspects"][aspect]["peer_median"]}


def _deterministic_draft(state: PlanState) -> dict:
    items = []
    for p in state["priorities"]:
        a = p["aspect"]
        ev = state["evidence"].get(a, [])
        mets = _aspect_metrics(state, a)
        themes = detect_sub_themes(a, [e["quote"] for e in ev])
        trend = "rising" if p["change"] > 0.02 else "persistent"
        headline = f"{a.capitalize()} complaints are {trend}" + (" (unusual spike)" if p["spike"] else "")
        metric_text = (f"{mets['complaint_rate_3m']:.0%} of reviews in the last 3 months complained about {a}, "
                       f"versus {mets['complaint_rate_prev6m']:.0%} in the six months before")
        if mets["peer_median"] == mets["peer_median"]:  # not NaN
            metric_text += f"; similar restaurants sit at {mets['peer_median']:.0%}"
        claims = [{"type": "metric", "text": metric_text + "."}]
        for e in ev:
            claims.append({"type": "review", "text": f"A {e['stars']:.0f}-star review on {e['date']} says: \"{e['quote']}\"",
                           "review_ids": [e["review_id"]], "quote": e["quote"]})
        items.append({"aspect": a, "headline": headline, "claims": claims, "actions": actions_for(a, themes),
                      "sub_themes": themes})
    return {"items": items}


def _llm_draft(state: PlanState) -> dict | None:
    """Ask Claude for a plan. Returns None on any failure so the caller can fall back."""
    import anthropic

    payload = {
        "restaurant": STORE.businesses.loc[state["business_id"], ["name", "city", "cuisine"]].to_dict(),
        "forecast": {k: state["forecast"][k] for k in ("risk", "risk_level", "attributions")},
        "priorities": state["priorities"],
        "metrics_by_aspect": {p["aspect"]: _aspect_metrics(state, p["aspect"]) for p in state["priorities"]},
        "evidence_reviews": {a: [{k: e[k] for k in ("review_id", "date", "stars", "text")} for e in ev]
                             for a, ev in state["evidence"].items()},
        "playbook": {p["aspect"]: PLAYBOOK[p["aspect"]] for p in state["priorities"]},
    }
    feedback = ""
    if state.get("feedback"):
        feedback = "\n\nYour previous draft had claims that failed verification:\n- " + "\n- ".join(state["feedback"])
    prompt = (
        "You are an operations advisor for a small restaurant. Using ONLY the data below, write a ranked action plan "
        "with one item per aspect in `priorities` (keep that order).\n"
        "Rules: every claim must be either (a) type 'review': cite review_ids from evidence_reviews for that aspect "
        "and include a short quote copied verbatim from one cited review, or (b) type 'metric': restate numbers from "
        "metrics_by_aspect (percentages rounded to whole numbers). Do not invent facts. Actions must be concrete, "
        "cheap and doable within two weeks; adapt them from the playbook.\n"
        "Return only JSON: {\"summary\": str, \"items\": [{\"aspect\": str, \"headline\": str, \"claims\": "
        "[{\"type\": \"review\"|\"metric\", \"text\": str, \"review_ids\": [str], \"quote\": str}], \"actions\": [str]}]}"
        f"{feedback}\n\nDATA:\n{json.dumps(payload, default=str)}"
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(model=LLM_MODEL, max_tokens=16000,
                                     messages=[{"role": "user", "content": prompt}])
        if msg.stop_reason == "refusal":
            return None
        raw = "".join(b.text for b in msg.content if b.type == "text")
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except Exception as exc:  # network, auth, parse errors -> deterministic fallback
        print(f"[M5] LLM draft failed ({type(exc).__name__}: {exc}); using deterministic planner")
        return None


def draft(state: PlanState) -> PlanState:
    mode, d = "deterministic", None
    if state.get("use_llm"):
        d = _llm_draft(state)
        mode = "llm" if d else "deterministic"
    if d is None:
        d = _deterministic_draft(state)
    pri = {p["aspect"]: p for p in state["priorities"]}
    d["items"] = [it for it in d.get("items", []) if it.get("aspect") in pri]
    for it in d["items"]:
        it["_metrics"] = _aspect_metrics(state, it["aspect"])
    d["business_id"] = state["business_id"]
    return {"draft_plan": d, "mode": mode, "attempts": state.get("attempts", 0) + 1}


def verify(state: PlanState) -> PlanState:
    checked = verify_plan(json.loads(json.dumps(state["draft_plan"], default=str)))
    failures = [f"[{it['aspect']}] {c['text'][:80]}... -> {'; '.join(c['reasons'])}"
                for it in checked["items"] for c in it["claims"] if not c["verified"]]
    return {"draft_plan": checked, "feedback": failures}


def route_after_verify(state: PlanState) -> str:
    if state.get("feedback") and state.get("mode") == "llm" and state["attempts"] < MAX_ATTEMPTS:
        return "draft"
    return "finalise"


def finalise(state: PlanState) -> PlanState:
    d, fc = state["draft_plan"], state["forecast"]
    pri = {p["aspect"]: p for p in state["priorities"]}
    biz = STORE.businesses.loc[state["business_id"]]
    items = []
    for rank, it in enumerate(sorted(d["items"], key=lambda i: -pri[i["aspect"]]["priority"]), start=1):
        p = pri[it["aspect"]]
        kept = [c for c in it["claims"] if c["verified"]]
        if not any(c.get("type") == "review" for c in kept) and it["aspect"] != state["priorities"][0]["aspect"]:
            continue  # an item without any verified review evidence is not shown (except the top watch item)
        items.append({
            "rank": rank, "aspect": it["aspect"], "priority": p["priority"], "headline": it.get("headline", ""),
            "why": (f"SHAP contribution to decline risk {p['shap']:+.2f} (logit), complaint z-score "
                    f"{p['robust_z']:.1f}, change in complaint rate {p['change']:+.0%}."),
            "claims": [{k: v for k, v in c.items() if k != "reasons"} for c in kept],
            "actions": it.get("actions", []), "sub_themes": it.get("sub_themes", []),
        })
    for i, it in enumerate(items, start=1):
        it["rank"] = i
    summary = d.get("summary") or (
        f"{biz['name']} has a {fc['risk']:.0%} estimated probability of its rating dropping by 0.3 stars or more over "
        f"the next 3 months ({fc['risk_level']} risk)."
        + (f" Address {items[0]['aspect']} first." if items else ""))
    plan = {"business_id": state["business_id"], "name": biz["name"], "month": fc["month"], "risk": fc["risk"],
            "risk_level": fc["risk_level"], "summary": summary, "items": items,
            "verification": d["verification"], "mode": state["mode"], "attempts": state["attempts"],
            "generated_at": datetime.now().isoformat(timespec="seconds")}
    return {"plan": plan}


def build_graph():
    g = StateGraph(PlanState)
    for name, fn in (("gather", gather), ("prioritise", prioritise), ("retrieve", retrieve), ("draft", draft),
                     ("verify", verify), ("finalise", finalise)):
        g.add_node(name, fn)
    g.set_entry_point("gather")
    g.add_edge("gather", "prioritise")
    g.add_edge("prioritise", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"draft": "draft", "finalise": "finalise"})
    g.add_edge("finalise", END)
    return g.compile()


_GRAPH = None


def generate_plan(business_id: str, use_llm: bool | None = None, save: bool = True) -> dict[str, Any]:
    """Run the agent for one restaurant and return the verified plan."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    if use_llm is None:
        use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    plan = _GRAPH.invoke({"business_id": business_id, "use_llm": use_llm})["plan"]
    if save:
        (PLANS_DIR / f"{business_id}.json").write_text(json.dumps(plan, indent=2, default=str))
    return plan


def run_batch(levels=("high", "medium"), use_llm: bool | None = None) -> dict:
    """Weekly job: generate plans for every at-risk restaurant and report groundedness."""
    latest = STORE.risk_latest
    active = STORE.features[(STORE.features["month"] == STORE.latest_month) & (STORE.features["n_reviews_3m"] > 0)]
    targets = latest[latest["risk_level"].isin(levels) & latest.index.isin(active["business_id"])].index
    total = verified = 0
    modes: dict[str, int] = {}
    for bid in targets:
        plan = generate_plan(bid, use_llm=use_llm)
        total += plan["verification"]["claims_total"]
        verified += plan["verification"]["claims_verified"]
        modes[plan["mode"]] = modes.get(plan["mode"], 0) + 1
    report = {"plans": len(targets), "claims_total": total, "claims_verified": verified,
              "groundedness": round(verified / total, 4) if total else None, "modes": modes}
    (OUTPUTS_DIR / "m5_evaluation.json").write_text(json.dumps(report, indent=2))
    print("[M5]", report)
    return report


if __name__ == "__main__":
    run_batch()
