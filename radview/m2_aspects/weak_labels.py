"""M2 - weak supervision.

Yelp has star ratings but no per-aspect labels. We bootstrap them with a
seed lexicon (which aspect a sentence is about) plus a polarity lexicon (how
the reviewer feels), and fall back on the star rating as a polarity signal
when the sentence itself carries no sentiment words:

    stars >= 4 -> positive,   stars <= 3 -> negative

The resulting labels are noisy; they are used to *train* the classifiers and
are themselves evaluated as the "lexicon rule" baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from radview.config import ASPECTS, LABEL_NEG, LABEL_NONE, LABEL_POS
from radview.lexicon import ASPECT_TERMS, sentence_polarity, split_sentences, tokenize


def label_review(text: str, stars: float) -> dict[str, int]:
    """Return {aspect: label} for one review."""
    pos = dict.fromkeys(ASPECTS, 0)
    neg = dict.fromkeys(ASPECTS, 0)
    for sent in split_sentences(text):
        toks = tokenize(sent)
        tokset = set(toks)
        hits = [a for a in ASPECTS if tokset & ASPECT_TERMS[a]]
        if not hits:
            continue
        pol = sentence_polarity(toks)
        if pol == 0:
            if stars is None or stars != stars:  # unknown rating (e.g. SemEval sentences)
                continue
            pol = 1 if stars >= 4 else -1
        for a in hits:
            if pol > 0:
                pos[a] += 1
            else:
                neg[a] += 1
    out = {}
    for a in ASPECTS:
        if neg[a] == 0 and pos[a] == 0:
            out[a] = LABEL_NONE
        elif neg[a] >= pos[a]:
            out[a] = LABEL_NEG
        else:
            out[a] = LABEL_POS
    return out


def weak_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised-ish wrapper: returns a frame with one int column per aspect."""
    rows = [label_review(t, s) for t, s in zip(df["text"].values, df["stars"].values)]
    return pd.DataFrame(rows, index=df.index)[ASPECTS].astype(np.int64)
