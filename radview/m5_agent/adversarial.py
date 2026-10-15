"""M5 - an adversarial planner that hallucinates on purpose.

A verifier that has only ever seen well-behaved plans has not been tested.
``HallucinatingPlanner`` produces one claim for every way an LLM can go wrong,
each of which the verifier must reject:

    1. cites a review id that does not exist                      -> citation
    2. cites a real review of *another* restaurant                -> citation
    3. cites a review of this restaurant that was never retrieved -> citation
    4. quotes text that is not in the cited review                -> quotation
    5. quotes a verbatim sentence that is praise, not a complaint -> relevance
    6. states a percentage no tool returned                       -> statistic
    7. states a star figure no tool returned                      -> statistic
    8. makes a "metric" claim with no checkable figure            -> substance

Run through the full agent graph, its plans must score groundedness 0.
"""
from __future__ import annotations

from radview.config import LABEL_POS
from radview.m5_agent.tools import STORE


class HallucinatingPlanner:
    name = "adversarial"

    def draft(self, state) -> tuple[dict, dict]:
        bid = state["business_id"]
        aspect = state["priorities"][0]["aspect"]
        rev = STORE.reviews
        mine = rev[rev["business_id"] == bid]
        other = rev[rev["business_id"] != bid].iloc[0]
        retrieved = set(state.get("retrieved_ids", []))
        not_retrieved = mine[~mine["review_id"].isin(retrieved)]
        unretrieved = not_retrieved.iloc[0] if len(not_retrieved) else mine.iloc[0]
        praised = mine[mine[f"{aspect}_label"] == LABEL_POS]
        any_mine = mine.iloc[0]

        claims = [
            {"type": "review", "text": "A guest reported food poisoning.", "review_ids": ["fake-review-000"],
             "quote": "I got food poisoning."},
            {"type": "review", "text": f"A guest wrote: \"{other['text'][:60]}\"", "review_ids": [other["review_id"]],
             "quote": other["text"][:60]},
            {"type": "review", "text": "An unretrieved review.", "review_ids": [unretrieved["review_id"]],
             "quote": unretrieved["text"].split(".")[0]},
            {"type": "review", "text": "A guest said the chef was rude.", "review_ids": [any_mine["review_id"]],
             "quote": "The chef shouted at us and threw a plate."},
            {"type": "metric", "text": f"87% of reviews complained about {aspect}."},
            {"type": "metric", "text": f"{aspect.capitalize()} problems cost this restaurant 1.75 stars."},
            {"type": "metric", "text": f"{aspect.capitalize()} is clearly the main problem."},
        ]
        if len(praised):
            r = praised.iloc[0]
            sentence = r["text"].split(".")[0]
            claims.append({"type": "review", "text": f"A guest complained: \"{sentence}\"",
                           "review_ids": [r["review_id"]], "quote": sentence})
        plan = {"summary": "Fabricated plan used to test the verifier.",
                "items": [{"aspect": aspect, "headline": "Invented problem", "claims": claims,
                           "actions": ["Do something drastic."]}]}
        return plan, {}
