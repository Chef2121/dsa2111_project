"""Seed lexicons used for weak supervision (M2) and evidence analysis (M5).

Generic words that appear in almost every review ("ordered", "dinner",
"menu") are deliberately excluded from the aspect lists: a sentence such as
"Ordered dinner for my birthday" says nothing about service or food quality,
and including them made weak labels copy the star rating onto it.

The aspect lexicon decides *which* aspect a sentence talks about; the polarity
lexicon decides *how* the reviewer feels about it. Both are deliberately small
and transparent: they only need to be good enough to bootstrap noisy training
labels that the transformer then generalises from.
"""
from __future__ import annotations

import re

ASPECT_TERMS: dict[str, set[str]] = {
    "food": {
        "food", "dish", "dishes", "meal", "meals", "taste", "tasted", "tastes", "flavor", "flavour",
        "flavors", "portion", "portions", "fries", "burger", "burgers", "pizza", "pasta",
        "chicken", "steak", "sauce", "dessert", "desserts", "cooked", "overcooked", "undercooked",
        "fresh", "stale", "soggy", "salty", "greasy", "bland", "delicious", "tasty", "seasoned",
        "appetizer", "appetizers", "entree", "entrees", "noodles", "soup", "salad", "tacos", "sushi",
        "chef", "kitchen", "ingredients",
    },
    "service": {
        "service", "staff", "waiter", "waitress", "server", "servers", "host", "hostess", "manager",
        "wait", "waited", "waiting", "rude", "friendly", "attentive", "bartender", "employees",
        "employee", "cashier", "welcoming", "polite", "ignored", "helpful",
    },
    "cleanliness": {
        "clean", "cleaner", "dirty", "filthy", "sticky", "bathroom", "bathrooms", "restroom",
        "restrooms", "hygiene", "hygienic", "spotless", "gross", "cockroach", "roach", "bug", "bugs",
        "fly", "flies", "smell", "smelled", "smelly", "grimy", "unsanitary", "tidy", "messy",
    },
    "price": {
        "price", "prices", "priced", "expensive", "cheap", "overpriced", "value", "worth", "cost",
        "costs", "bill", "pricey", "affordable", "budget", "deal", "money", "charge", "charged",
    },
    "ambience": {
        "ambience", "ambiance", "atmosphere", "decor", "music", "noisy", "loud", "cozy", "cosy",
        "vibe", "vibes", "lighting", "seating", "crowded", "quiet", "interior", "patio", "romantic",
        "cramped", "setting",
    },
}

POSITIVE_TERMS = {
    "good", "great", "amazing", "excellent", "delicious", "friendly", "attentive", "clean", "spotless",
    "cozy", "fresh", "perfect", "love", "loved", "best", "reasonable", "affordable", "worth",
    "fantastic", "tasty", "nice", "pleasant", "quick", "fast", "awesome", "wonderful", "welcoming",
    "polite", "helpful", "tidy", "lovely", "superb", "outstanding", "generous", "relaxed", "charming",
    "flavorful", "impeccable", "fair",
}

NEGATIVE_TERMS = {
    "bad", "terrible", "awful", "bland", "cold", "rude", "slow", "dirty", "filthy", "sticky",
    "overpriced", "expensive", "noisy", "loud", "stale", "soggy", "worst", "disappointing",
    "disappointed", "gross", "horrible", "greasy", "burnt", "ignored", "mediocre", "cramped",
    "unsanitary", "smelly", "grimy", "pricey", "dismissive", "careless", "wrong", "messy",
    "overcooked", "undercooked", "lukewarm", "inedible", "unfriendly", "chaotic", "poor", "rip-off",
}

NEGATORS = {"not", "no", "never", "hardly", "wasn't", "isn't", "weren't", "didn't", "don't", "nothing"}

# Sub-themes used by the M5 action planner to make recommendations specific.
SUB_THEMES: dict[str, dict[str, set[str]]] = {
    "food": {
        "temperature": {"cold", "lukewarm", "warm"},
        "freshness": {"stale", "soggy", "fresh", "old", "wilted"},
        "doneness": {"overcooked", "undercooked", "burnt", "raw", "dry"},
        "seasoning": {"bland", "salty", "seasoned", "flavor", "flavour", "taste"},
        "portions": {"portion", "portions", "small", "tiny"},
    },
    "service": {
        "wait times": {"wait", "waited", "waiting", "slow", "minutes", "forever", "took"},
        "staff attitude": {"rude", "dismissive", "unfriendly", "attitude", "ignored", "eye"},
        "order accuracy": {"wrong", "forgot", "missing", "mistake", "mixed", "received", "lost", "incorrect"},
    },
    "cleanliness": {
        "restrooms": {"bathroom", "bathrooms", "restroom", "restrooms"},
        "dining area": {"table", "tables", "floor", "booth", "sticky", "crumbs", "tacky"},
        "pests & odours": {"roach", "cockroach", "fly", "flies", "bug", "bugs", "smell", "smelled"},
    },
    "price": {
        "value for money": {"worth", "value", "portion", "portions", "steep"},
        "price increases": {"increase", "increased", "raised", "went", "higher", "hike"},
        "hidden charges": {"charge", "charged", "fee", "fees", "surcharge"},
    },
    "ambience": {
        "noise": {"loud", "noisy", "music", "hear", "shout"},
        "comfort & space": {"cramped", "crowded", "seating", "seats", "chairs", "tight"},
        "temperature & lighting": {"lighting", "dark", "hot", "freezing", "stuffy"},
    },
}

_TOKEN_RE = re.compile(r"[a-z]+(?:['-][a-z]+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens (keeps contractions such as ``wasn't``)."""
    return _TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    """Very light sentence splitter - good enough for review text."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def sentence_polarity(tokens: list[str]) -> int:
    """Return +1 / -1 / 0 using lexicon hits with a 3-token negation window."""
    score = 0
    for i, tok in enumerate(tokens):
        val = 1 if tok in POSITIVE_TERMS else -1 if tok in NEGATIVE_TERMS else 0
        if val == 0:
            continue
        if any(t in NEGATORS or t.endswith("n't") for t in tokens[max(0, i - 3):i]):
            val = -val
        score += val
    return (score > 0) - (score < 0)
