"""M5 command line: generate one plan, or compare planners.

    python -m radview.m5_agent.cli --business-id <id>            # one plan (Claude if a key is set)
    python -m radview.m5_agent.cli --compare 20                  # planners on the 20 riskiest restaurants

The comparison runs the rule-based planner, the Claude planner (only when
ANTHROPIC_API_KEY is set - never silently substituted) and the adversarial
planner, and reports groundedness, per-axis verifier failures, items per plan,
latency and token use -> outputs/m5_planner_comparison.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from radview.config import OUTPUTS_DIR
from radview.m5_agent.agent import at_risk_restaurants, generate_plan
from radview.m5_agent.tools import STORE


def compare(n: int) -> dict:
    targets = at_risk_restaurants()[:n] or list(STORE.risk_latest.sort_values("risk", ascending=False).index[:n])
    planners = ["rule_based", "adversarial"] + (["claude"] if os.getenv("ANTHROPIC_API_KEY") else [])
    report = {"restaurants": len(targets), "planners": {}}
    for name in planners:
        rows = [generate_plan(b, planner=name, save=False) for b in targets]
        total = sum(p["verification"]["claims_total"] for p in rows)
        ok = sum(p["verification"]["claims_verified"] for p in rows)
        axes: dict[str, int] = {}
        for p in rows:
            for k, v in p["verification"]["failures_by_axis"].items():
                axes[k] = axes.get(k, 0) + v
        tokens = [p["llm_usage"]["input_tokens"] + p["llm_usage"]["output_tokens"] for p in rows if p.get("llm_usage")]
        report["planners"][name] = {
            "actually_ran": sorted({p["planner"] for p in rows}),
            "claims_total": total, "claims_verified": ok, "groundedness": round(ok / total, 4) if total else None,
            "failures_by_axis": axes,
            "mean_items_per_plan": round(statistics.mean(len(p["items"]) for p in rows), 2) if rows else None,
            "median_latency_s": statistics.median(p["latency_s"] for p in rows) if rows else None,
            "mean_tool_calls": round(statistics.mean(len(p["tool_calls"]) for p in rows), 1) if rows else None,
            "mean_tokens": round(statistics.mean(tokens)) if tokens else None,
        }
        print(f"[M5 compare] {name:12s} groundedness {report['planners'][name]['groundedness']}  "
              f"failures {axes}")
    if "claude" not in planners:
        report["note"] = "Claude planner not run: ANTHROPIC_API_KEY is not set."
    (OUTPUTS_DIR / "m5_planner_comparison.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--business-id")
    ap.add_argument("--compare", type=int, metavar="N", help="compare planners on the N riskiest restaurants")
    ap.add_argument("--planner", choices=["rule_based", "claude", "adversarial"])
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        bid = args.business_id or STORE.risk_latest.sort_values("risk").index[-1]
        print(json.dumps(generate_plan(bid, planner=args.planner), indent=2, default=str))
