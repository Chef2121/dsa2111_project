"""Write action plans as anonymised text for the usefulness-rating form.

Each plan is shown as an owner would see it on the dashboard (summary, then each item's evidence and actions),
with the restaurant's name replaced by "Restaurant N" and only its cuisine kept. Only verified claims are
printed, so every quote and figure in the output has passed the verifier.

    # rule-based plans, from the saved plans of the last pipeline run
    python scripts/export_form_plans.py --out form_plans.md BID1 BID2 ...
    # open-model (Qwen) plans: needs a running Ollama server with qwen2.5:7b (RADVIEW_LLM_PROVIDER=open)
    python scripts/export_form_plans.py --planner open_model --out form_plans_qwen.md BID1 BID2 ...

The output contains Yelp review text: keep it inside the group and the form.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/export_form_plans.py`
from radview.m5_agent.agent import PLANS_DIR, generate_plan  # noqa: E402
from radview.m5_agent.tools import STORE  # noqa: E402


def plan_text(plan: dict, label: str) -> str:
    name = STORE.businesses.loc[plan["business_id"], "name"]
    lines = [plan["summary"].replace(name, label), ""]
    for it in plan["items"]:
        lines.append(f"{it['rank']}. {it['aspect'].upper()}: {it['headline']}")
        lines.append("   Evidence:")
        lines += [f"   • {c['text']}" for c in it["claims"]] or ["   • (no verified evidence)"]
        lines.append("   What to do:")
        lines += [f"   • {a}" for a in it["actions"]] or ["   • (no actions given)"]
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("business_ids", nargs="+")
    ap.add_argument("--planner", default="saved", choices=["saved", "rule_based", "open_model", "claude"],
                    help="'saved' reads the plans of the last pipeline run; the others generate a fresh plan")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    parts, raw = [], {}
    for i, bid in enumerate(args.business_ids, start=1):
        if args.planner == "saved":
            plan = json.loads((PLANS_DIR / f"{bid}.json").read_text())
        else:
            plan = generate_plan(bid, planner=args.planner, save=False)
        raw[bid] = plan
        cuisine = STORE.businesses.loc[bid, "cuisine"]
        parts.append(f"## Restaurant {i} ({cuisine}, Philadelphia) [{plan['planner']}]\n\n"
                     f"{plan_text(plan, f'Restaurant {i}')}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")  # full record
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
