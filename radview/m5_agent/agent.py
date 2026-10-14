"""M5 - grounded agentic prioritisation (LangGraph).

Workflow (one run per restaurant)::

    gather ──> prioritise ──> retrieve ──> draft ──> verify ──┬──> finalise
                                             ^                │
                                             └── (redraft) ───┘  Claude planner only, max 2 attempts

* gather      - calls the analytics tools: query_metrics, run_forecast, compare_peers
* prioritise  - ranks aspects by impact (SHAP share + counterfactual risk drop),
                urgency (complaint spike z-score, growth) and the effort the fix needs
* retrieve    - get_evidence_reviews for each prioritised aspect
* draft       - one of two planners behind the same interface:
                  ClaudePlanner     Claude runs its own tool-calling loop (it may call any
                                    tool again, e.g. fetch evidence for another aspect),
                                    then submits a structured plan via ``submit_plan``
                  RuleBasedPlanner  deterministic; the offline fallback *and* the baseline
                                    the LLM is measured against
* verify      - mechanical claim checks (verifier.py); only reviews retrieved in this run
                may be cited
* finalise    - unverified claims are dropped, groundedness and per-axis failures reported

The plan always records which planner produced it, so a fallback can never be
mistaken for an LLM plan.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from radview.config import ASPECTS, LLM_MODEL, OUTPUTS_DIR
from radview.m5_agent.playbook import PLAYBOOK, actions_for, detect_sub_themes
from radview.m5_agent.tools import (STORE, TOOL_SCHEMAS, compare_peers, execute_tool, get_evidence_reviews,
                                    query_metrics, run_forecast)
from radview.m5_agent.verifier import verify_plan

MAX_ITEMS = 3
MAX_ATTEMPTS = 2
MAX_TOOL_TURNS = 10
PLANS_DIR = OUTPUTS_DIR / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

# Relative effort to act on each aspect (1 = typical). Assumption, stated in the report:
# cleanliness fixes are quick and cheap; ambience usually needs capital spending.
EFFORT = {"food": 1.0, "service": 0.8, "cleanliness": 0.6, "price": 0.9, "ambience": 1.3}


class PlanState(TypedDict, total=False):
    business_id: str
    use_llm: bool
    planner_override: str   # "rule_based" | "claude" | "adversarial" (evaluation); default: by use_llm
    metrics: dict
    forecast: dict
    peers: dict
    priorities: list[dict]
    evidence: dict[str, list[dict]]
    retrieved_ids: list[str]
    tool_calls: list[dict]
    draft_plan: dict
    attempts: int
    feedback: list[str]
    planner: str
    llm_usage: dict
    plan: dict


# --------------------------------------------------------------------------
# Analytics nodes
# --------------------------------------------------------------------------
def gather(state: PlanState) -> PlanState:
    bid = state["business_id"]
    calls = [{"tool": t, "args": {}} for t in ("query_metrics", "run_forecast", "compare_peers")]
    return {"metrics": query_metrics(bid), "forecast": run_forecast(bid), "peers": compare_peers(bid),
            "attempts": 0, "tool_calls": calls, "retrieved_ids": []}


def _share(values: dict[str, float]) -> dict[str, float]:
    pos = {a: max(values.get(a, 0.0), 0.0) for a in ASPECTS}
    total = sum(pos.values())
    return {a: (v / total if total > 0 else 0.0) for a, v in pos.items()}


def prioritise(state: PlanState) -> PlanState:
    """priority = (0.35*SHAP share + 0.35*counterfactual share + 0.30*urgency) / effort"""
    fc, aspects = state["forecast"], state["metrics"]["aspects"]
    shap_share = _share(fc["attributions"])
    cf_share = _share(fc.get("counterfactual_risk_drop", {}))
    ranked = []
    for a in ASPECTS:
        m = aspects[a]
        urgency = 0.6 * min(max(m["robust_z"], 0.0) / 5.0, 1.0) + 0.4 * min(max(m["change"], 0.0) / 0.2, 1.0)
        impact = 0.5 * shap_share[a] + 0.5 * cf_share[a]
        score = (0.7 * impact + 0.3 * urgency) / EFFORT[a]
        ranked.append({"aspect": a, "priority": round(score, 4), "impact": round(impact, 4),
                       "urgency": round(urgency, 4), "effort": EFFORT[a],
                       "shap": round(fc["attributions"].get(a, 0.0), 4),
                       "risk_drop_if_fixed": round(fc.get("counterfactual_risk_drop", {}).get(a, 0.0), 4),
                       "star_cost": round(fc.get("star_cost", {}).get(a, 0.0), 3),
                       "robust_z": m["robust_z"], "change": m["change"], "spike": m["spike"],
                       "complaint_rate_3m": m["complaint_rate_3m"]})
    ranked.sort(key=lambda r: -r["priority"])
    chosen = [r for r in ranked if r["priority"] > 0.05 and r["complaint_rate_3m"] > 0][:MAX_ITEMS]
    if not chosen:  # nothing pressing: surface the most-complained-about aspect as a watch item
        chosen = sorted(ranked, key=lambda r: -r["complaint_rate_3m"])[:1]
    return {"priorities": chosen}


def retrieve(state: PlanState) -> PlanState:
    bid = state["business_id"]
    evidence = {p["aspect"]: get_evidence_reviews(bid, p["aspect"]) for p in state["priorities"]}
    ids = sorted({e["review_id"] for ev in evidence.values() for e in ev})
    calls = state["tool_calls"] + [{"tool": "get_evidence_reviews", "args": {"aspect": a}} for a in evidence]
    return {"evidence": evidence, "retrieved_ids": ids, "tool_calls": calls}


def aspect_metrics(state: PlanState, aspect: str) -> dict:
    """The figures a claim about ``aspect`` may cite (rates as fractions, star values as stars)."""
    m = state["metrics"]["aspects"][aspect]
    return {"rates": {"complaint_rate_3m": m["complaint_rate_3m"], "complaint_rate_prev6m": m["complaint_rate_prev6m"],
                      "praise_rate_3m": m["praise_rate_3m"], "market_median_rate": m.get("market_median_rate"),
                      "peer_median": state["peers"]["aspects"][aspect]["peer_median"],
                      "risk_drop_if_fixed": state["forecast"].get("counterfactual_risk_drop", {}).get(aspect),
                      "risk": state["forecast"]["risk"]},
            "stars": {"star_cost": state["forecast"].get("star_cost", {}).get(aspect),
                      "rating_3m": state["metrics"]["rating_3m"], "rating_6m": state["metrics"]["rating_6m"]}}


# --------------------------------------------------------------------------
# Planners
# --------------------------------------------------------------------------
class RuleBasedPlanner:
    """Deterministic plan from templates + the playbook. No API key, no network."""
    name = "rule_based"

    def draft(self, state: PlanState) -> tuple[dict, dict]:
        items = []
        for p in state["priorities"]:
            a = p["aspect"]
            ev = state["evidence"].get(a, [])
            mets = aspect_metrics(state, a)
            rates, stars = mets["rates"], mets["stars"]
            themes = detect_sub_themes(a, [e["quote"] for e in ev])
            trend = "rising" if p["change"] > 0.02 else "persistent"
            headline = f"{a.capitalize()} complaints are {trend}" + (" (unusual spike)" if p["spike"] else "")
            text = (f"{rates['complaint_rate_3m']:.0%} of reviews in the last 3 months complained about {a}, "
                    f"versus {rates['complaint_rate_prev6m']:.0%} in the six months before")
            if rates["market_median_rate"] is not None:
                text += f"; the typical restaurant in your market sits at {rates['market_median_rate']:.0%}"
            claims = [{"type": "metric", "text": text + "."}]
            if (stars["star_cost"] or 0) >= 0.01:
                claims.append({"type": "metric", "text": f"Compared with the typical restaurant in your market, "
                               f"{a} complaints are estimated to cost about {stars['star_cost']:.2f} stars."})
            for e in ev:
                claims.append({"type": "review", "text": f"A {e['stars']:.0f}-star review on {e['date']} says: \"{e['quote']}\"",
                               "review_ids": [e["review_id"]], "quote": e["quote"]})
            items.append({"aspect": a, "headline": headline, "claims": claims, "actions": actions_for(a, themes),
                          "sub_themes": themes})
        return {"items": items}, {}


SUBMIT_PLAN_TOOL = {
    "name": "submit_plan",
    "description": "Submit the final ranked action plan. Call exactly once, after gathering evidence.",
    "strict": True,
    "input_schema": {
        "type": "object", "additionalProperties": False, "required": ["summary", "items"],
        "properties": {
            "summary": {"type": "string", "description": "Two sentences for the owner: the risk and what to fix first."},
            "items": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["aspect", "headline", "claims", "actions"],
                "properties": {
                    "aspect": {"type": "string", "enum": ASPECTS},
                    "headline": {"type": "string"},
                    "claims": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["type", "text", "review_ids", "quote"],
                        "properties": {"type": {"type": "string", "enum": ["review", "metric"]},
                                       "text": {"type": "string"},
                                       "review_ids": {"type": "array", "items": {"type": "string"}},
                                       "quote": {"type": "string", "description": "verbatim sentence, '' for metric claims"}}}},
                    "actions": {"type": "array", "items": {"type": "string"}}}}}}},
}

SYSTEM_PROMPT = """You are an operations advisor for a small, independent restaurant. You write a short,
ranked action plan the owner can act on within two weeks.

You can only know what the tools tell you. Investigate with the tools (they are already scoped to this
restaurant), then call submit_plan exactly once.

Rules for the plan:
- One item per aspect worth fixing, most important first. The analytics have pre-ranked the aspects; you
  may reorder them only if the tool data clearly supports it.
- Every claim is either type "review" or type "metric".
  * "review": cite review_ids returned by get_evidence_reviews for that aspect, and copy one sentence from a
    cited review verbatim into "quote". Never cite a review you have not retrieved.
  * "metric": restate figures from the tools. Percentages as whole numbers (e.g. 23%); star values with two
    decimals (e.g. 0.12 stars). Set review_ids to [] and quote to "".
- Do not state any number the tools did not return. Do not invent reviews or quotes.
- Actions must be concrete and cheap; adapt them from the playbook provided.
Every claim is checked mechanically afterwards; unsupported claims are deleted before the owner sees the plan."""


class ClaudePlanner:
    """Claude with tool calling. The loop is written out (not delegated to an SDK helper) because every
    evidence id the model sees must be recorded - the verifier checks citations against them."""
    name = "claude"

    def __init__(self, model: str = LLM_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def draft(self, state: PlanState) -> tuple[dict | None, dict]:
        bid = state["business_id"]
        biz = STORE.businesses.loc[bid]
        brief = {"restaurant": {"name": biz["name"], "cuisine": biz["cuisine"], "city": biz["city"]},
                 "pre_ranked_priorities": state["priorities"],
                 "playbook": {p["aspect"]: PLAYBOOK[p["aspect"]] for p in state["priorities"]}}
        text = ("Write the action plan for this restaurant.\n" + json.dumps(brief, default=str))
        if state.get("feedback"):
            text += ("\n\nYour previous plan had claims that failed verification - fix or drop them:\n- "
                     + "\n- ".join(state["feedback"]))
        messages: list[dict] = [{"role": "user", "content": text}]
        tools = TOOL_SCHEMAS + [SUBMIT_PLAN_TOOL]
        retrieved, calls = set(), []
        usage = {"input_tokens": 0, "output_tokens": 0, "turns": 0}
        for _ in range(MAX_TOOL_TURNS):
            resp = self.client.messages.create(model=self.model, max_tokens=16000, system=SYSTEM_PROMPT,
                                               tools=tools, messages=messages)
            usage["turns"] += 1
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                return None, usage
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                return None, usage  # ended without submitting a plan
            results, plan = [], None
            for tu in uses:
                calls.append({"tool": tu.name, "args": tu.input})
                if tu.name == "submit_plan":
                    plan = tu.input
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "received"})
                    continue
                try:
                    out = execute_tool(bid, tu.name, tu.input)
                    if tu.name == "get_evidence_reviews":
                        retrieved |= {r["review_id"] for r in out["reviews"]}
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(out, default=str)})
                except Exception as exc:  # report tool errors back to the model instead of crashing
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": str(exc), "is_error": True})
            if plan is not None:
                for it in plan.get("items", []):
                    for c in it.get("claims", []):
                        c["quote"] = c.get("quote") or None
                return plan, {"usage": usage, "retrieved": retrieved, "calls": calls}
            messages.append({"role": "user", "content": results})
        return None, usage


# --------------------------------------------------------------------------
# Draft / verify / finalise nodes
# --------------------------------------------------------------------------
def draft(state: PlanState) -> PlanState:
    plan, extra, planner = None, {}, "rule_based"
    if state.get("planner_override") == "adversarial":
        from radview.m5_agent.adversarial import HallucinatingPlanner
        plan, extra = HallucinatingPlanner().draft(state)
        planner = "adversarial"
    elif state.get("use_llm"):
        try:
            plan, extra = ClaudePlanner().draft(state)
            planner = "claude"
        except Exception as exc:  # no credentials, network, API errors -> deterministic fallback
            print(f"[M5] Claude planner unavailable ({type(exc).__name__}: {exc}); using rule-based planner")
        if plan is None:
            planner = "rule_based"
    if plan is None:
        plan, extra = RuleBasedPlanner().draft(state)
    pri = {p["aspect"]: p for p in state["priorities"]}
    plan["items"] = [it for it in plan.get("items", []) if it.get("aspect") in pri]
    for it in plan["items"]:
        it["_metrics"] = aspect_metrics(state, it["aspect"])
    plan["business_id"] = state["business_id"]
    out: PlanState = {"draft_plan": plan, "planner": planner, "attempts": state.get("attempts", 0) + 1}
    if extra.get("retrieved"):
        out["retrieved_ids"] = sorted(set(state["retrieved_ids"]) | extra["retrieved"])
    if extra.get("calls"):
        out["tool_calls"] = state["tool_calls"] + extra["calls"]
    if extra.get("usage"):
        out["llm_usage"] = extra["usage"]
    return out


def verify(state: PlanState) -> PlanState:
    checked = verify_plan(json.loads(json.dumps(state["draft_plan"], default=str)),
                          retrieved_ids=set(state["retrieved_ids"]))
    failures = [f"[{it['aspect']}] {c['text'][:80]}... -> {'; '.join(c['reasons'])}"
                for it in checked["items"] for c in it["claims"] if not c["verified"]]
    return {"draft_plan": checked, "feedback": failures}


def route_after_verify(state: PlanState) -> str:
    if state.get("feedback") and state.get("planner") == "claude" and state["attempts"] < MAX_ATTEMPTS:
        return "draft"
    return "finalise"


def finalise(state: PlanState) -> PlanState:
    d, fc = state["draft_plan"], state["forecast"]
    pri = {p["aspect"]: p for p in state["priorities"]}
    biz = STORE.businesses.loc[state["business_id"]]
    items = []
    for it in sorted(d["items"], key=lambda i: -pri[i["aspect"]]["priority"]):
        p = pri[it["aspect"]]
        kept = [c for c in it["claims"] if c["verified"]]
        if not any(c.get("type") == "review" for c in kept) and it["aspect"] != state["priorities"][0]["aspect"]:
            continue  # an item without any verified review evidence is not shown (except the top watch item)
        items.append({
            "aspect": it["aspect"], "priority": p["priority"], "headline": it.get("headline", ""),
            "risk_drop_if_fixed": p["risk_drop_if_fixed"], "star_cost": p["star_cost"], "effort": p["effort"],
            "why": (f"Fixing it would lower decline risk by about {p['risk_drop_if_fixed']:.0%} (counterfactual); "
                    f"SHAP contribution {p['shap']:+.2f}; complaint z-score {p['robust_z']:.1f}; "
                    f"complaint rate change {p['change']:+.0%}."),
            "claims": [{k: v for k, v in c.items() if k not in ("reasons", "failed_axes")} for c in kept],
            "actions": it.get("actions", []), "sub_themes": it.get("sub_themes", []),
        })
    for i, it in enumerate(items, start=1):
        it["rank"] = i
    summary = d.get("summary") or (
        f"{biz['name']} has a {fc['risk']:.0%} estimated probability of its rating dropping by 0.3 stars or more over "
        f"the next 3 months ({fc['risk_level']} risk)." + (f" Address {items[0]['aspect']} first." if items else ""))
    rejected = [{"aspect": it["aspect"], "text": c["text"], "reasons": c["reasons"]}
                for it in d["items"] for c in it["claims"] if not c["verified"]]
    plan = {"business_id": state["business_id"], "name": biz["name"], "month": fc["month"], "risk": fc["risk"],
            "risk_level": fc["risk_level"], "summary": summary, "items": items,
            "verification": d["verification"], "rejected_claims": rejected,
            "planner": state["planner"], "mode": "llm" if state["planner"] == "claude" else "deterministic",
            "attempts": state["attempts"], "tool_calls": state["tool_calls"], "llm_usage": state.get("llm_usage"),
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


def generate_plan(business_id: str, use_llm: bool | None = None, save: bool = True,
                  planner: str | None = None) -> dict[str, Any]:
    """Run the agent for one restaurant and return the verified plan.

    ``planner`` forces a planner for evaluation ("rule_based", "claude", "adversarial").
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    if use_llm is None:
        use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    t0 = time.time()
    if planner == "claude":
        use_llm = True
    elif planner == "rule_based":
        use_llm = False
    plan = _GRAPH.invoke({"business_id": business_id, "use_llm": use_llm,
                          "planner_override": planner or ""})["plan"]
    plan["latency_s"] = round(time.time() - t0, 2)
    if save:
        (PLANS_DIR / f"{business_id}.json").write_text(json.dumps(plan, indent=2, default=str))
    return plan


def at_risk_restaurants(levels=("high", "medium")) -> list[str]:
    latest = STORE.risk_latest
    active = STORE.features[(STORE.features["month"] == STORE.latest_month) & (STORE.features["n_reviews_3m"] > 0)]
    return list(latest[latest["risk_level"].isin(levels) & latest.index.isin(active["business_id"])].index)


def run_batch(levels=("high", "medium"), use_llm: bool | None = None) -> dict:
    """Weekly job: generate plans for every at-risk restaurant and report groundedness."""
    targets = at_risk_restaurants(levels)
    total = verified = 0
    planners: dict[str, int] = {}
    axes: dict[str, int] = {}
    latencies = []
    for bid in targets:
        plan = generate_plan(bid, use_llm=use_llm)
        v = plan["verification"]
        total += v["claims_total"]
        verified += v["claims_verified"]
        planners[plan["planner"]] = planners.get(plan["planner"], 0) + 1
        for k, n in v["failures_by_axis"].items():
            axes[k] = axes.get(k, 0) + n
        latencies.append(plan["latency_s"])
    report = {"plans": len(targets), "claims_total": total, "claims_verified": verified,
              "groundedness": round(verified / total, 4) if total else None, "planners": planners,
              "modes": planners, "failures_by_axis": axes,
              "median_latency_s": round(float(sorted(latencies)[len(latencies) // 2]), 2) if latencies else None}
    (OUTPUTS_DIR / "m5_evaluation.json").write_text(json.dumps(report, indent=2))
    print("[M5]", report)
    return report


if __name__ == "__main__":
    run_batch()
