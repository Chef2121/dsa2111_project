"""M2 - aspect-level sentiment classifiers.

All models expose the same interface::

    model.predict_proba(texts) -> np.ndarray of shape (n, len(ASPECTS), 3)

where the last axis is (not mentioned, positive, negative). This lets the
evaluation harness and the scoring job swap models freely.

* ``LexiconModel``     - the weak-labelling rule itself (baseline 1)
* ``TfidfModel``       - TF-IDF + one logistic regression per aspect (baseline 2)
* ``TransformerModel`` - DistilBERT encoder with one 3-way head per aspect
                         fine-tuned on the weak labels (proposed model)
* ``ZeroShotLLMModel`` - zero-shot LLM prompt (baseline 3, needs an API key)
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np

from radview.config import ASPECTS, LLM_MODEL, MAX_SEQ_LEN, MODELS_DIR, RANDOM_SEED, TRANSFORMER_BASE
from radview.m2_aspects.weak_labels import label_review

N_CLS = 3


def _onehot(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, N_CLS), dtype=np.float32)
    np.put_along_axis(out, labels[..., None], 1.0, axis=-1)
    return out


class LexiconModel:
    name = "lexicon_rule"

    def predict_proba(self, texts, stars=None):
        stars = stars if stars is not None else [float("nan")] * len(texts)
        labs = np.array([[label_review(t, s)[a] for a in ASPECTS] for t, s in zip(texts, stars)])
        return _onehot(labs)


class TfidfModel:
    """TF-IDF (word 1-2 grams) shared across aspects + a multinomial LR per aspect."""
    name = "tfidf_logreg"

    def __init__(self):
        self.vec = None
        self.clfs = {}

    def fit(self, texts, y: np.ndarray):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        for i, a in enumerate(ASPECTS):
            clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
            clf.fit(X, y[:, i])
            self.clfs[a] = clf
        return self

    def predict_proba(self, texts, stars=None):
        X = self.vec.transform(texts)
        out = np.zeros((X.shape[0], len(ASPECTS), N_CLS), dtype=np.float32)
        for i, a in enumerate(ASPECTS):
            clf = self.clfs[a]
            out[:, i, clf.classes_] = clf.predict_proba(X)
        return out

    def save(self, path: Path = MODELS_DIR / "tfidf_aspects.joblib"):
        joblib.dump({"vec": self.vec, "clfs": self.clfs}, path)

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "tfidf_aspects.joblib"):
        obj = cls()
        d = joblib.load(path)
        obj.vec, obj.clfs = d["vec"], d["clfs"]
        return obj


class TransformerModel:
    """Fine-tuned DistilBERT with a multi-head (aspect x polarity) classifier."""
    name = "distilbert_finetuned"
    DIR = MODELS_DIR / "transformer_aspects"

    def __init__(self, base: str = TRANSFORMER_BASE):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(base)
        self.encoder = AutoModel.from_pretrained(base)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(ASPECTS) * N_CLS)
        self.encoder.to(self.device)
        self.head.to(self.device)

    def _forward(self, batch_texts, encoder=None):
        enc = self.tok(list(batch_texts), truncation=True, max_length=MAX_SEQ_LEN, padding=True,
                       return_tensors="pt").to(self.device)
        hidden = (encoder or self.encoder)(**enc).last_hidden_state[:, 0]  # [CLS] representation
        return self.head(hidden).view(-1, len(ASPECTS), N_CLS)

    def fit(self, texts, y: np.ndarray, epochs: int = 2, batch_size: int = 32, lr: float = 3e-5):
        torch = self.torch
        torch.manual_seed(RANDOM_SEED)
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        n = len(texts)
        steps = epochs * ((n + batch_size - 1) // batch_size)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
        # Class weights counter the dominance of "not mentioned" for rare aspects.
        counts = np.stack([np.bincount(y[:, i], minlength=N_CLS) for i in range(len(ASPECTS))]) + 1
        w = torch.tensor((counts.sum(1, keepdims=True) / (N_CLS * counts)) ** 0.5, dtype=torch.float32, device=self.device)
        texts = np.asarray(texts, dtype=object)
        y_t = torch.tensor(y, dtype=torch.long)
        rng = np.random.default_rng(RANDOM_SEED)
        self.encoder.train()
        step, t0 = 0, time.time()
        for ep in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                logits = self._forward(texts[idx])
                yb = y_t[idx].to(self.device)
                loss = sum(torch.nn.functional.cross_entropy(logits[:, i], yb[:, i], weight=w[i])
                           for i in range(len(ASPECTS)))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 50 == 0:
                    print(f"  [transformer] epoch {ep + 1} step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
        self.encoder.eval()
        return self

    def _inference_encoder(self):
        """int8 dynamic quantisation of the Linear layers: ~2x faster on CPU, negligible accuracy loss."""
        if self.device != "cpu":
            return self.encoder
        if getattr(self, "_qencoder", None) is None:
            self._qencoder = self.torch.ao.quantization.quantize_dynamic(
                copy.deepcopy(self.encoder).eval(), {self.torch.nn.Linear}, dtype=self.torch.qint8)
        return self._qencoder

    def predict_proba(self, texts, stars=None, batch_size: int = 128, quantize: bool = True):
        torch = self.torch
        self.encoder.eval()
        encoder = self._inference_encoder() if quantize else self.encoder
        texts = list(texts)
        # sort by length so padding is minimal, then restore order
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(ASPECTS), N_CLS), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                idx = order[s:s + batch_size]
                logits = self._forward([texts[i] for i in idx], encoder)
                out[idx] = torch.softmax(logits, -1).cpu().numpy()
        return out

    def save(self, path: Path = DIR):
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        self.tok.save_pretrained(path / "encoder")
        self.torch.save(self.head.state_dict(), path / "head.pt")

    @classmethod
    def load(cls, path: Path = DIR):
        obj = cls(base=str(path / "encoder"))
        obj.head.load_state_dict(obj.torch.load(path / "head.pt", map_location=obj.device))
        obj.encoder.eval()
        return obj


class ZeroShotLLMModel:
    """Zero-shot baseline: ask an LLM to label each review (requires ANTHROPIC_API_KEY)."""
    name = "zero_shot_llm"
    PROMPT = (
        "Label the restaurant review below for each aspect: food, service, cleanliness, price, ambience. "
        "For each aspect answer exactly one of: none (not discussed), positive, negative. "
        "Mild complaints count as negative. Respond with only a JSON object, e.g. "
        '{{"food": "positive", "service": "none", "cleanliness": "none", "price": "negative", "ambience": "none"}}.\n\n'
        "Review: {text}"
    )

    def __init__(self, model: str = LLM_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def predict_proba(self, texts, stars=None):
        mapping = {"none": 0, "positive": 1, "negative": 2}
        labs = []
        for t in texts:
            msg = self.client.messages.create(model=self.model, max_tokens=2000,
                                              messages=[{"role": "user", "content": self.PROMPT.format(text=t)}])
            raw = "" if msg.stop_reason == "refusal" else "".join(b.text for b in msg.content if b.type == "text")
            try:
                d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            except ValueError:
                d = {}
            labs.append([mapping.get(str(d.get(a, "none")).lower(), 0) for a in ASPECTS])
        return _onehot(np.array(labs))


def load_best_model():
    """Load the model chosen by the M2 evaluation (falls back to TF-IDF)."""
    choice_file = MODELS_DIR / "m2_selected.json"
    choice = json.loads(choice_file.read_text())["model"] if choice_file.exists() else "tfidf_logreg"
    if choice == TransformerModel.name and (TransformerModel.DIR / "head.pt").exists():
        return TransformerModel.load()
    return TfidfModel.load()
