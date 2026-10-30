"""Agent tests on the pipeline outputs of the active data source (skipped if absent)."""
from __future__ import annotations

import json

import pytest

from radview.config import OUTPUTS_DIR

pytestmark = pytest.mark.skipif(not (OUTPUTS_DIR / "risk_latest.parquet").exists(),
                                reason="run `python -m radview.pipeline` first")


@pytest.fixture(scope="module")
def riskiest():
    """The restaurant with the highest predicted risk: the agent's most important case."""
    from radview.m5_agent.tools import STORE
    return STORE.risk_latest.sort_values("risk").index[-1]


def test_rule_based_plan_is_fully_grounded(riskiest):
    """The deterministic planner's plan passes the verifier in full, its items are ranked 1..n and each has at
    least one action.
    """
    from radview.m5_agent.agent import generate_plan

    plan = generate_plan(riskiest, planner="rule_based", save=False)
    assert plan["planner"] == "rule_based"
    assert plan["verification"]["groundedness"] == 1.0
    assert [it["rank"] for it in plan["items"]] == list(range(1, len(plan["items"]) + 1))
    for it in plan["items"]:
        assert it["actions"], "every item needs at least one action"


def test_hallucinating_planner_is_completely_rejected(riskiest):
    """Every fabricated claim must fail, and none may reach the owner."""
    from radview.m5_agent.agent import generate_plan

    plan = generate_plan(riskiest, planner="adversarial", save=False)
    v = plan["verification"]
    assert v["claims_total"] >= 7 and v["claims_verified"] == 0 and v["groundedness"] == 0
    for axis in ("citation", "quotation", "statistic", "substance"):
        assert v["failures_by_axis"][axis] > 0, axis
    assert all(not it["claims"] for it in plan["items"])
    assert len(plan["rejected_claims"]) == v["claims_total"]


def test_claude_planner_falls_back_without_credentials(riskiest, monkeypatch):
    """Without Claude credentials, an LLM plan request falls back to the rule-based planner instead of failing."""
    from radview.m5_agent.agent import generate_plan

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    plan = generate_plan(riskiest, use_llm=True, save=False)
    assert plan["planner"] in {"rule_based", "claude"}  # "claude" only if another credential source exists
    assert plan["verification"]["groundedness"] in (None, 1.0) or plan["planner"] == "claude"


# ---------------------------------------------------------------- free open-model planner
class _FakeResponse:
    """Minimal stand-in for a requests.Response from an OpenAI-compatible chat server."""
    ok, status_code, text = True, 200, ""

    def __init__(self, payload):
        """Wrap the JSON ``payload`` the fake server returns."""
        self.payload = payload

    def raise_for_status(self):
        """Never raises: tests signal an error through ``ok`` / ``status_code`` instead."""
        pass

    def json(self):
        """Return the stored payload."""
        return self.payload


def _reply(name: str, args: dict) -> _FakeResponse:
    """An OpenAI-compatible chat response containing one tool call (arguments as a JSON string)."""
    call = {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    return _FakeResponse({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]}}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20}})


def test_open_model_planner_uses_tools_and_is_verified(monkeypatch):
    """A scripted Ollama-style server: the model fetches evidence itself, then cites one real review and
    invents another. The real claim must pass and the invented one must be removed."""
    from radview.config import OPEN_LLM_MODEL
    from radview.m5_agent import open_llm
    from radview.m5_agent.agent import at_risk_restaurants, generate_plan

    # the riskiest restaurant that has citable review evidence for one of its priorities
    for bid in at_risk_restaurants():
        rule = generate_plan(bid, planner="rule_based", save=False)
        aspects = [it["aspect"] for it in rule["items"] if any(c["type"] == "review" for c in it["claims"])]
        if aspects:
            break
    else:
        pytest.skip("no at-risk restaurant has review evidence")
    aspect = aspects[0]

    def fake_post(url, **kw):
        """Scripted server: first ask for evidence, then submit one real and one invented review claim."""
        tool_msgs = [m for m in kw["json"]["messages"] if m["role"] == "tool"]
        if not tool_msgs:
            return _reply("get_evidence_reviews", {"aspect": aspect})
        ev = json.loads(tool_msgs[-1]["content"])["reviews"][0]
        claims = [{"type": "review", "text": f'A review says: "{ev["quote"]}"', "review_ids": [ev["review_id"]],
                   "quote": ev["quote"]},
                  {"type": "review", "text": 'A review says: "The chef shouted at us."',
                   "review_ids": ["invented-review-id"], "quote": "The chef shouted at us."}]
        return _reply("submit_plan", {"summary": "Fix this first.", "items": [
            {"aspect": aspect, "headline": "h", "claims": claims, "actions": ["a"]}]})

    monkeypatch.setattr(open_llm.requests, "post", fake_post)
    plan = generate_plan(bid, planner="open_model", save=False)
    assert plan["planner"] == "open_model" and plan["mode"] == "llm"
    assert plan["llm_usage"]["model"] == OPEN_LLM_MODEL and plan["llm_usage"]["turns"] == 2
    assert {"tool": "get_evidence_reviews", "args": {"aspect": aspect}} in plan["tool_calls"]
    v = plan["verification"]
    assert (v["claims_total"], v["claims_verified"]) == (2, 1) and v["failures_by_axis"]["citation"] >= 1
    shown = [c for it in plan["items"] for c in it["claims"]]
    assert all("invented-review-id" not in c.get("review_ids", []) for c in shown)
    assert plan["attempts"] == 2  # the failed claim triggered one redraft


def test_open_model_planner_retries_a_stuck_generation(riskiest, monkeypatch):
    """Ollama aborts greedy repetition loops with a 500; one sampled retry must recover the plan."""
    from radview.m5_agent import open_llm
    from radview.m5_agent.agent import generate_plan

    temperatures = []

    def flaky_post(url, **kw):
        """Fail the first request like Ollama's repetition abort (HTTP 500), then answer normally; records each
        temperature.
        """
        temperatures.append(kw["json"]["temperature"])
        if len(temperatures) == 1:
            resp = _FakeResponse({"error": {"message": "prediction aborted, token repeat limit reached"}})
            resp.ok, resp.status_code, resp.text = False, 500, "token repeat limit reached"
            return resp
        return _reply("submit_plan", {"summary": "s", "items": []})

    monkeypatch.setattr(open_llm.requests, "post", flaky_post)
    plan = generate_plan(riskiest, planner="open_model", save=False)
    assert plan["planner"] == "open_model" and plan["llm_usage"]["retries"] >= 1
    assert temperatures[:2] == [0.0, open_llm.RETRY_TEMPERATURE]


def test_open_model_planner_falls_back_when_server_is_down(riskiest, monkeypatch):
    """If the open-model server cannot be reached, the agent falls back to the rule-based planner and still returns
    a fully grounded plan.
    """
    import requests

    from radview.m5_agent import open_llm
    from radview.m5_agent.agent import generate_plan

    def refuse(url, **kw):
        """Simulate a server that refuses the connection."""
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(open_llm.requests, "post", refuse)
    plan = generate_plan(riskiest, planner="open_model", save=False)
    assert plan["planner"] == "rule_based" and plan["verification"]["groundedness"] == 1.0
