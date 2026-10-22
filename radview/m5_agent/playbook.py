"""Operational playbook: concrete fixes per aspect and sub-theme.

Used by the deterministic planner and given to the LLM as grounding so its
recommendations stay practical for a small restaurant.
"""
from __future__ import annotations

from radview.lexicon import SUB_THEMES, aspect_tokens

PLAYBOOK: dict[str, dict[str, list[str]]] = {
    "food": {
        "temperature": ["Audit pass-to-table time at peak; add a heat lamp or expeditor so plates leave within 60 s.",
                        "Pre-warm plates for hot dishes."],
        "freshness": ["Review FIFO stock rotation and prep par levels so items are not held too long.",
                      "Fry/prepare crisp items to order rather than batching."],
        "doneness": ["Re-train line cooks on cooking specs and use probe thermometers for proteins.",
                     "Add a chef quality check at the pass during the busiest two hours."],
        "seasoning": ["Taste-test signature dishes before each service against a written recipe card.",
                      "Check whether any supplier or recipe changed recently."],
        "portions": ["Re-publish plating specs with photos and weigh portions during spot checks."],
        "_default": ["Hold a kitchen quality review on the most-complained-about dishes.",
                     "Ask the head chef to taste and sign off the top 5 dishes daily for two weeks."],
    },
    "service": {
        "wait times": ["Check staffing against covers for the slowest shifts; add a floater at peak times.",
                       "Set a greet-within-2-minutes standard and track ticket times."],
        "staff attitude": ["Run a short hospitality refresher and recognise staff named positively in reviews.",
                           "Have the manager do table touches during every peak service."],
        "order accuracy": ["Read back orders and use the POS modifier fields instead of free text.",
                           "Add an expo check of each ticket before food leaves the kitchen."],
        "_default": ["Review service-related complaints with the floor team in the next pre-shift meeting."],
    },
    "cleanliness": {
        "restrooms": ["Introduce a 30-minute restroom check log signed by staff.",
                      "Restock soap/paper towels at every check."],
        "dining area": ["Wipe-down and floor sweep between every table turn; spot-check before opening."],
        "pests & odours": ["Book a pest-control inspection this week and seal entry points.",
                           "Deep-clean drains and bins; review waste pickup frequency."],
        "_default": ["Do a full deep clean and introduce an opening/closing cleanliness checklist."],
    },
    "price": {
        "value for money": ["Re-balance portion size vs. price on the top-selling dishes or add a value set.",
                            "Make premium ingredients visible on the menu to justify price."],
        "price increases": ["Communicate recent price changes (menu note) and keep a few lower-priced options.",
                            "Consider a loyalty or lunch special to retain regulars."],
        "hidden charges": ["Show all service charges and add-ons clearly on the menu and bill."],
        "_default": ["Benchmark prices against nearby peers of the same cuisine."],
    },
    "ambience": {
        "noise": ["Lower background music volume at peak and add soft furnishings to absorb sound."],
        "comfort & space": ["Review table layout; remove one or two tables to ease crowding."],
        "temperature & lighting": ["Check HVAC settings during service and adjust lighting levels."],
        "_default": ["Walk the dining room as a guest during peak hours and note comfort issues."],
    },
}


def detect_sub_themes(aspect: str, texts: list[str]) -> list[str]:
    """Sub-themes (e.g. 'wait times') mentioned in the evidence, most frequent first."""
    counts: dict[str, int] = {}
    for t in texts:
        toks = set(aspect_tokens(t))  # "can't wait to come back" is not a wait-time complaint
        for theme, words in SUB_THEMES.get(aspect, {}).items():
            if toks & words:
                counts[theme] = counts.get(theme, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def actions_for(aspect: str, themes: list[str], max_actions: int = 3) -> list[str]:
    book = PLAYBOOK[aspect]
    acts: list[str] = []
    for th in themes:
        acts.extend(book.get(th, []))
    acts.extend(book["_default"])
    seen, out = set(), []
    for a in acts:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out[:max_actions]
