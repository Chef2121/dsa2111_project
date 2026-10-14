"""Agent tests on the pipeline outputs of the active data source (skipped if absent)."""
from __future__ import annotations

import pytest

from radview.config import OUTPUTS_DIR

pytestmark = pytest.mark.skipif(not (OUTPUTS_DIR / "risk_latest.parquet").exists(),
                                reason="run `python -m radview.pipeline` first")


@pytest.fixture(scope="module")
def riskiest():
    from radview.m5_agent.tools import STORE
    return STORE.risk_latest.sort_values("risk").index[-1]


def test_rule_based_plan_is_fully_grounded(riskiest):
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
    from radview.m5_agent.agent import generate_plan

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    plan = generate_plan(riskiest, use_llm=True, save=False)
    assert plan["planner"] in {"rule_based", "claude"}  # "claude" only if another credential source exists
    assert plan["verification"]["groundedness"] in (None, 1.0) or plan["planner"] == "claude"
