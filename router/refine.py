"""
refine.py — Train router v1 and compare it with v0 on held-out Eval-1000.

Usage:
    python -m router.refine              # train v1, score v0 / v1 variants once
    python -m router.refine --select     # also re-run both selection steps

Protocol (Eval-1000 is never used to choose anything):
    1. Model selection by 5x repeated stratified 5-fold CV on the 239 seed
       examples only (--select reproduces it). Picked by CV accuracy, then
       log-loss: word unigrams + character 2-5-grams, C=16. Versus v0's
       config, CV accuracy 0.947 vs 0.936 and log-loss 0.172 vs 0.349 — the
       accuracy gain is noise-sized, the calibration gain is not, and the
       cascade threshold (0.65) is a probability, so calibration matters.
    2. Training data: seed only, and seed + data/distilled_queries.jsonl
       (router/distill.py: Gemma-written queries, Gemma-classifier labels,
       near-duplicates of Eval-1000 removed). The distilled rows are all
       length counter-examples, so at full weight they swamp the seed data.
       Their sample weight and C were chosen on a dev set — Eval-500 minus
       its overlaps with Eval-1000 and the seed data (438 queries): weight
       0.1, C=2 (dev accuracy 0.984 vs 0.966 with no distilled data).
    3. Every variant is scored once on Eval-1000. The cascade is scored with
       the cached real classifier verdicts in data/classifier_verdicts_1000.jsonl
       (router/live_cascade_eval.py --all), at the unchanged 0.65 threshold.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from config import ML_ROUTER_CONFIDENCE_THRESHOLD
from router.eval_data import load_cases
from router.features import encode_text
from router.ml_router import MLRouter
from router.seed_data import load_seed

HERE = os.path.dirname(__file__)
V0_PATH = os.path.join(HERE, "models", "router_v0.joblib")
V1_PATH = os.path.join(HERE, "models", "router_v1.joblib")
DISTILLED_WEIGHT = 0.1          # chosen on the Eval-500 dev set, see docstring
DISTILLED_C = 2.0
DISTILLED_PATH = os.path.join(HERE, "..", "data", "distilled_queries.jsonl")
VERDICTS_PATH = os.path.join(HERE, "..", "data", "classifier_verdicts_1000.jsonl")
RESULTS_PATH = os.path.join(HERE, "refine_results.json")
WORD_RULE_MAX = 8
SEED = 0


def build_v1(C: float = 16.0, word_ngrams: tuple[int, int] = (1, 1), char: bool = True) -> Pipeline:
    feats: list = [("word", TfidfVectorizer(ngram_range=word_ngrams, sublinear_tf=True,
                                            max_features=5000))]
    if char:
        feats.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                              sublinear_tf=True, max_features=20000)))
    return Pipeline([
        ("features", FeatureUnion(feats)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000)),
    ])


def select(seed_data) -> None:
    """The CV grid behind build_v1's defaults (seed data only)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

    X = [encode_text(q) for q, _ in seed_data]
    y = [lbl for _, lbl in seed_data]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    for ng, char, C, calib in itertools.product(
            [(1, 1), (1, 2), (1, 3)], [False, True], [0.5, 1, 2, 4, 8, 16], [False, True]):
        pipe = build_v1(C, ng, char)
        if calib:
            pipe.steps[-1] = ("clf", CalibratedClassifierCV(pipe.steps[-1][1], cv=5, method="sigmoid"))
        r = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "neg_log_loss"])
        rows.append((r["test_accuracy"].mean(), -r["test_neg_log_loss"].mean(), ng, char, C, calib))
    rows.sort(key=lambda r: (-r[0], r[1]))
    print("CV model selection (seed data only), top 5:")
    for acc, ll, ng, char, C, calib in rows[:5]:
        print(f"  acc {acc:.3f}  log-loss {ll:.3f}  word{ng} char={char} C={C} calibrated={calib}")


def _norm(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def fit(seed_data, distilled, weight: float, C: float) -> Pipeline:
    data = seed_data + (distilled if weight > 0 else [])
    sw = np.array([1.0] * len(seed_data) + [weight] * (len(data) - len(seed_data)))
    return build_v1(C=C).fit([encode_text(q) for q, _ in data], [lbl for _, lbl in data],
                             clf__sample_weight=sw)


def select_weight(seed_data, distilled) -> None:
    """Distilled-row weight and C, chosen on Eval-500 minus overlaps (dev)."""
    held = {_norm(q) for q, _, _ in load_cases("eval_1000.jsonl")} | {_norm(q) for q, _ in seed_data}
    dev = [(q, lbl) for q, lbl, _ in load_cases("eval_500.jsonl") if _norm(q) not in held]
    X, y = [encode_text(q) for q, _ in dev], np.array([lbl for _, lbl in dev])
    print(f"Distilled weight selection on dev ({len(dev)} queries from Eval-500):")
    for C in (2.0, 16.0):
        for w in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            acc = (fit(seed_data, distilled, w, C).predict(X) == y).mean()
            print(f"  C={C:<4} weight={w:<5} dev accuracy {acc:.3f}")


def bootstrap_ci(correct: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = rng.choice(correct, size=(n, len(correct))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(name: str, decisions, confidences, cases, verdicts) -> dict:
    truth = np.array([gt for _, gt, _ in cases])
    dec = np.array(decisions)
    conf = np.array(confidences)
    deferred = conf < ML_ROUTER_CONFIDENCE_THRESHOLD
    cascade = np.where(deferred, [verdicts[q] for q, _, _ in cases], dec)
    lo, hi = bootstrap_ci((cascade == truth).astype(float))

    words = np.array([len(re.findall(r"\w+", q)) for q, _, _ in cases])
    rule_wrong = np.where(words <= WORD_RULE_MAX, "small", "big") != truth
    return {
        "router": name,
        "ml_accuracy": float((dec == truth).mean()),
        "deferred_share": float(deferred.mean()),
        "cascade_accuracy": float((cascade == truth).mean()),
        "cascade_ci": [lo, hi],
        "large_share": float((cascade == "big").mean()),
        # the 103 queries where length alone misleads — the real test
        "cascade_acc_where_length_rule_wrong": float((cascade == truth)[rule_wrong].mean()),
        "n_length_rule_wrong": int(rule_wrong.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--select", action="store_true", help="re-run the CV model selection")
    args = ap.parse_args()

    seed = load_seed()
    if args.select:
        select(seed)

    cases = load_cases("eval_1000.jsonl")
    with open(VERDICTS_PATH, encoding="utf-8") as f:
        verdicts = {r["query"]: r["final"] for r in map(json.loads, f)}
    distilled = []
    if os.path.exists(DISTILLED_PATH):
        with open(DISTILLED_PATH, encoding="utf-8") as f:
            distilled = [(r["query"], r["label"]) for r in map(json.loads, f) if r["parsed"]]

    def predict(router: MLRouter):
        preds = [router.predict(q) for q, _, _ in cases]
        return [p.decision for p in preds], [p.confidence for p in preds]

    if args.select and distilled:
        select_weight(seed, distilled)

    variants = [
        ("v0 (published)", MLRouter.load(V0_PATH)),
        ("v1, seed only", MLRouter(pipeline=fit(seed, [], 0.0, 16.0))),
    ]
    if distilled:
        variants += [
            ("v1, + distilled @1.0", MLRouter(pipeline=fit(seed, distilled, 1.0, 16.0))),
            (f"v1, + distilled @{DISTILLED_WEIGHT}",
             MLRouter(pipeline=fit(seed, distilled, DISTILLED_WEIGHT, DISTILLED_C))),
        ]

    results = [score(name, *predict(r), cases, verdicts) for name, r in variants]

    words = np.array([len(re.findall(r"\w+", q)) for q, _, _ in cases])
    truth = np.array([gt for _, gt, _ in cases])
    rule_acc = float((np.where(words <= WORD_RULE_MAX, "small", "big") == truth).mean())
    print(f"\nEval-1000 (held out). Word-count rule: {rule_acc:.1%}. "
          f"Cascade uses cached Gemma 4 E4B verdicts at threshold {ML_ROUTER_CONFIDENCE_THRESHOLD}.")
    if distilled:
        print(f"Distilled training rows: {len(distilled)} "
              f"({sum(lbl == 'big' for _, lbl in distilled) / len(distilled):.0%} big)")
    print(f"\n{'Router':<24}{'ML acc':>8}{'Deferred':>10}{'Cascade':>9}{'95% CI':>17}"
          f"{'Large %':>9}{'Len-rule-wrong':>16}")
    print("─" * 93)
    for r in results:
        ci = f"[{r['cascade_ci'][0]:.3f},{r['cascade_ci'][1]:.3f}]"
        print(f"{r['router']:<24}{r['ml_accuracy']:>8.1%}{r['deferred_share']:>10.1%}"
              f"{r['cascade_accuracy']:>9.1%}{ci:>17}{r['large_share']:>9.1%}"
              f"{r['cascade_acc_where_length_rule_wrong']:>16.1%}")
    print(f"\n'Len-rule-wrong' = the {results[0]['n_length_rule_wrong']} queries where "
          f"≤8 words → small gets it wrong.")

    best_name, best_router = variants[-1]            # fixed in advance, not picked on Eval-1000
    best_router.save(V1_PATH)
    print(f"\nSaved {best_name} to {os.path.normpath(V1_PATH)}")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"word_rule_accuracy": rule_acc, "n_distilled": len(distilled),
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
