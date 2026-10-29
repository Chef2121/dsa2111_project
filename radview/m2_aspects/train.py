"""M2 - train, evaluate and apply the aspect-sentiment classifiers.

Steps
1. Load the human-labelled benchmarks:
   * ``SemEval-2014 Task 4`` restaurant test set (800 sentences, human labels for
     food / service / price / ambience - no cleanliness category);
   * ``labels/<source>/hand_labelled.csv`` - in-domain review labels. With
     synthetic data this is sampled from the simulator's ground truth; with
     real Yelp data the team fills in the exported ``to_label.csv``.
2. Weak-label a training sample (lexicon + star polarity, never touching the
   in-domain test reviews).
3. Train TF-IDF+LR (baseline) and fine-tune DistilBERT in PyTorch (proposed).
4. Evaluate every model (incl. the lexicon rule and, if an API key is set,
   a zero-shot LLM) with macro-F1 on each benchmark.
5. Select the best model on a *development* set - the in-domain labels, else 800
   sentences of the SemEval-2014 training file - so the reported SemEval test
   score is not also the score the model was chosen on - and score every review
   -> processed/review_aspects.parquet

Usage:  python -m radview.m2_aspects.train [--no-transformer] [--train-size 8000]
        RADVIEW_LLM_PROVIDER=open python -m radview.m2_aspects.train --zero-shot-only --zero-shot-n 800
                                       # add only the zero-shot LLM baseline to an existing evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from radview.config import (ASPECTS, LABEL_NEG, LABEL_NONE, LABEL_POS, LABELS_DIR, MODELS_DIR, OUTPUTS_DIR,
                            PROCESSED_DIR, RANDOM_SEED, RAW_DIR, SEMEVAL_FILE, SEMEVAL_TRAIN_FILE)
from radview.m2_aspects.models import (LexiconModel, TfidfModel, TransformerModel, ZeroShotLLMModel,
                                       load_best_model)
from radview.m2_aspects.weak_labels import weak_label_frame

HAND_LABELLED = LABELS_DIR / "hand_labelled.csv"
TEST_SIZE = 1000
DEV_SIZE = 800  # SemEval training sentences used only to choose the production model
DEV_BENCHMARK = "semeval2014_dev"
# Sentences per benchmark for the zero-shot LLM: small by default to bound API cost; the free local model can
# label a whole benchmark (--zero-shot-n 800), which makes it directly comparable with the other models.
ZERO_SHOT_N = int(os.getenv("RADVIEW_ZEROSHOT_N", "150"))
SEMEVAL_ASPECTS = ["food", "service", "price", "ambience"]  # SemEval-2014 has no cleanliness category
# SemEval polarities -> our 3-class scheme. "conflict" contains a complaint, so it counts as negative;
# "neutral" expresses no opinion, so it is treated as not-mentioned.
SEMEVAL_POLARITY = {"positive": LABEL_POS, "negative": LABEL_NEG, "conflict": LABEL_NEG, "neutral": LABEL_NONE}


def load_semeval(path=SEMEVAL_FILE) -> pd.DataFrame | None:
    """Parse a SemEval-2014 restaurant XML file into one row per sentence."""
    if not path.exists():
        return None
    rows = []
    for sent in ET.parse(path).getroot().iter("sentence"):
        row = {"review_id": sent.get("id"), "stars": np.nan, "text": sent.findtext("text"), **dict.fromkeys(ASPECTS, 0)}
        for cat in sent.iter("aspectCategory"):
            if cat.get("category") in SEMEVAL_ASPECTS:
                row[cat.get("category")] = SEMEVAL_POLARITY[cat.get("polarity")]
        rows.append(row)
    return pd.DataFrame(rows)


def load_in_domain(reviews: pd.DataFrame) -> pd.DataFrame | None:
    """Hand-labelled in-domain reviews (created from ground truth for synthetic data)."""
    if HAND_LABELLED.exists():
        df = pd.read_csv(HAND_LABELLED).dropna(subset=ASPECTS)
        return df if len(df) else None
    gt_path = RAW_DIR / "_ground_truth_aspects.jsonl"
    if gt_path.exists():
        gt = pd.read_json(gt_path, lines=True)
        test = reviews.merge(gt, on="review_id").sample(TEST_SIZE, random_state=RANDOM_SEED)
        test = test[["review_id", "stars", "text", *ASPECTS]].assign(label_source="simulated_annotation")
        test.to_csv(HAND_LABELLED, index=False)
        return test
    # Real data that nobody has labelled yet: export a sheet for the team.
    sheet_path = LABELS_DIR / "to_label.csv"
    if not sheet_path.exists():
        sheet = reviews.sample(min(300, len(reviews)), random_state=RANDOM_SEED)[["review_id", "stars", "text"]]
        for a in ASPECTS:
            sheet[a] = ""
        sheet.to_csv(sheet_path, index=False)
        print(f"[M2] exported {sheet_path} - label each aspect 0/1/2 (none/positive/negative), "
              "save as hand_labelled.csv and re-run to add an in-domain benchmark")
    return None


def evaluate(model, test: pd.DataFrame, aspects=ASPECTS) -> dict:
    """Macro-F1 (3 classes) per aspect + F1 for detecting complaints (negative class)."""
    t0 = time.time()
    proba = model.predict_proba(test["text"].tolist(), test["stars"].tolist())
    pred = proba.argmax(-1)
    y = test[ASPECTS].to_numpy().astype(int)
    per_aspect, neg_f1 = {}, {}
    for i, a in enumerate(ASPECTS):
        if a not in aspects:
            continue
        per_aspect[a] = round(f1_score(y[:, i], pred[:, i], average="macro", labels=[0, 1, 2], zero_division=0), 4)
        neg_f1[a] = round(f1_score(y[:, i] == LABEL_NEG, pred[:, i] == LABEL_NEG, zero_division=0), 4)
    return {
        "model": model.name, "n_test": len(test),
        "macro_f1": round(float(np.mean(list(per_aspect.values()))), 4),
        "complaint_f1": round(float(np.mean(list(neg_f1.values()))), 4),
        "macro_f1_per_aspect": per_aspect, "complaint_f1_per_aspect": neg_f1,
        "inference_ms_per_review": round(1000 * (time.time() - t0) / len(test), 2),
    }


def load_benchmarks(reviews: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Every labelled benchmark available: {name: (frame, aspects it covers)}."""
    benchmarks = {}
    in_domain = load_in_domain(reviews)
    if in_domain is not None:
        benchmarks["in_domain"] = (in_domain, ASPECTS)
    semeval = load_semeval()
    if semeval is not None:
        benchmarks["semeval2014"] = (semeval, SEMEVAL_ASPECTS)
    dev = load_semeval(SEMEVAL_TRAIN_FILE)
    if dev is not None:
        dev = dev.sample(min(DEV_SIZE, len(dev)), random_state=RANDOM_SEED).reset_index(drop=True)
        benchmarks[DEV_BENCHMARK] = (dev, SEMEVAL_ASPECTS)
    if not benchmarks:
        raise SystemExit("No labelled benchmark available (SemEval file missing and no hand labels).")
    return benchmarks


def evaluate_zero_shot(llm, df: pd.DataFrame, aspects: list[str], n: int = ZERO_SHOT_N) -> dict:
    """Evaluate the zero-shot LLM on a random sample of ``n`` texts (it is slow), recording the provider,
    model and how many replies were not valid JSON.
    """
    # a random (not first-n) sample, so a partial run is still a fair sample of the benchmark
    sample = df if n >= len(df) else df.sample(n, random_state=RANDOM_SEED)
    res = evaluate(llm, sample, aspects)
    res.update({"llm_provider": llm.provider, "llm_model": llm.model, "parse_failures": llm.parse_failures})
    return res


def run_zero_shot_only(n: int = ZERO_SHOT_N) -> dict:
    """Evaluate only the zero-shot LLM baseline and merge it into outputs/m2_evaluation.json.

    Every other result and the production model choice stay untouched, so this can run after a full M2 run -
    e.g. on a GPU machine with the free open-weight model (RADVIEW_LLM_PROVIDER=open).
    """
    path = OUTPUTS_DIR / "m2_evaluation.json"
    report = json.loads(path.read_text())
    llm = ZeroShotLLMModel()
    for name, (df, aspects) in load_benchmarks(pd.read_parquet(PROCESSED_DIR / "reviews.parquet")).items():
        if name == DEV_BENCHMARK:  # the LLM is a baseline, never a selection candidate
            continue
        print(f"[M2] zero-shot {llm.provider}/{llm.model} on {min(n, len(df))} {name} texts ...")
        res = evaluate_zero_shot(llm, df, aspects, n)
        print(f"[M2] zero_shot_llm {name:12s} macro-F1={res['macro_f1']:.3f} complaint-F1={res['complaint_f1']:.3f} "
              f"parse failures={res['parse_failures']}")
        report["results"][name] = [r for r in report["results"].get(name, []) if r["model"] != llm.name] + [res]
    path.write_text(json.dumps(report, indent=2))
    return report


def run(use_transformer: bool = True, train_size: int = 6000, tfidf_size: int = 40000, epochs: int = 2,
        reuse_transformer: bool = False) -> dict:
    """M2 entry point: weak-label the reviews, train TF-IDF and (optionally) DistilBERT, evaluate every model on
    every benchmark, and choose the production model on the selection set (never the test set if a
    dev set exists). Writes models/m2_selected.json and outputs/m2_evaluation.json.
    """
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
    benchmarks = load_benchmarks(reviews)
    in_domain = benchmarks.get("in_domain", (None,))[0]
    # Model selection uses the in-domain set when we have one, otherwise the SemEval dev set. Only if neither
    # exists does it fall back to the SemEval test set - and the report then says so (selected_on).
    select_on = next(b for b in ("in_domain", DEV_BENCHMARK, "semeval2014") if b in benchmarks)

    held_out = set(in_domain["review_id"]) if in_domain is not None else set()
    pool = reviews[~reviews["review_id"].isin(held_out)]
    print(f"[M2] weak-labelling {min(tfidf_size, len(pool))} training reviews ...")
    train = pool.sample(min(tfidf_size, len(pool)), random_state=RANDOM_SEED)
    y_weak = weak_label_frame(train).to_numpy()
    weak_dist = {a: np.bincount(y_weak[:, i], minlength=3).tolist() for i, a in enumerate(ASPECTS)}

    results: dict[str, list] = {name: [] for name in benchmarks}

    def record(model, extra=None):
        """Evaluate ``model`` on every benchmark and append the results (plus ``extra``) to the report."""
        for name, (df, aspects) in benchmarks.items():
            res = evaluate(model, df, aspects)
            res.update(extra or {})
            results[name].append(res)
            print(f"[M2] {model.name:22s} {name:12s} macro-F1={res['macro_f1']:.3f} complaint-F1={res['complaint_f1']:.3f}")

    record(LexiconModel())
    tfidf = TfidfModel().fit(train["text"].tolist(), y_weak)
    tfidf.save()
    record(tfidf)

    if use_transformer and reuse_transformer and (TransformerModel.DIR / "head.pt").exists():
        print("[M2] re-using saved fine-tuned transformer")
        record(TransformerModel.load())
    elif use_transformer:
        n = min(train_size, len(train))
        print(f"[M2] fine-tuning transformer on {n} weakly-labelled reviews ({epochs} epochs) ...")
        t0 = time.time()
        tr = TransformerModel().fit(train["text"].tolist()[:n], y_weak[:n], epochs=epochs)
        tr.save()
        record(tr, {"train_seconds": round(time.time() - t0, 1), "train_size": n})

    if ZeroShotLLMModel.available():
        try:
            llm = ZeroShotLLMModel()
            for name, (df, aspects) in benchmarks.items():
                if name != DEV_BENCHMARK:  # a baseline, never a selection candidate: no need to pay for dev
                    results[name].append(evaluate_zero_shot(llm, df, aspects))
        except Exception as exc:  # network / auth problems should not kill the pipeline
            print("[M2] zero-shot LLM skipped:", exc)
    else:
        for name in benchmarks:
            if name != DEV_BENCHMARK:
                results[name].append({"model": "zero_shot_llm",
                                      "skipped": "ANTHROPIC_API_KEY not set (or set RADVIEW_LLM_PROVIDER=open)"})

    candidates = [r for r in results[select_on] if r["model"] in {"tfidf_logreg", "distilbert_finetuned"}]
    best = max(candidates, key=lambda r: r["macro_f1"])["model"]
    (MODELS_DIR / "m2_selected.json").write_text(json.dumps({"model": best, "selected_on": select_on}))
    report = {"selected_model": best, "selected_on": select_on, "weak_label_distribution": weak_dist,
              "benchmarks": {k: {"n": len(v[0]), "aspects": v[1]} for k, v in benchmarks.items()},
              "results": results}
    (OUTPUTS_DIR / "m2_evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"[M2] selected model: {best} (on {select_on})")
    score_all(reviews)
    return report


def score_all(reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the selected model to every review and persist per-aspect predictions."""
    if reviews is None:
        reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
    model = load_best_model()
    print(f"[M2] scoring {len(reviews)} reviews with {model.name} ...")
    t0 = time.time()
    proba = model.predict_proba(reviews["text"].tolist(), reviews["stars"].tolist())
    out = pd.DataFrame({"review_id": reviews["review_id"].values})
    for i, a in enumerate(ASPECTS):
        out[f"{a}_label"] = proba[:, i].argmax(-1).astype(np.int8)
        out[f"{a}_neg_p"] = proba[:, i, LABEL_NEG].round(4)
        out[f"{a}_pos_p"] = proba[:, i, LABEL_POS].round(4)
    out.to_parquet(PROCESSED_DIR / "review_aspects.parquet", index=False)
    print(f"[M2] scored in {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-transformer", action="store_true", help="skip DistilBERT fine-tuning (fast mode)")
    ap.add_argument("--train-size", type=int, default=6000, help="same default as the pipeline")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--reuse-transformer", action="store_true", help="evaluate the saved model instead of re-training")
    ap.add_argument("--zero-shot-only", action="store_true",
                    help="evaluate only the zero-shot LLM baseline and add it to outputs/m2_evaluation.json")
    ap.add_argument("--zero-shot-n", type=int, default=ZERO_SHOT_N, help="texts per benchmark for the zero-shot LLM")
    args = ap.parse_args()
    if args.zero_shot_only:
        run_zero_shot_only(args.zero_shot_n)
    elif args.score_only:
        score_all()
    else:
        run(use_transformer=not args.no_transformer, train_size=args.train_size, epochs=args.epochs,
            reuse_transformer=args.reuse_transformer)
