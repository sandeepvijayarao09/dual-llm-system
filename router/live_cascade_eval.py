"""
live_cascade_eval.py — End-to-end routing accuracy with the real LLM classifier.

Usage:
    python -m router.live_cascade_eval          # the queries the ML router defers on
    python -m router.live_cascade_eval --all    # also: LLM classifier alone on all 1000

The offline evals (eval_1000, cost_eval) have to assume what the LLM
classifier decides for the ~15% of queries the ML router is unsure about.
This script asks it. Each deferred Eval-1000 query goes to the configured
Small LLM through modules.classifier, and the orchestrator's rule is applied
(a "small" verdict below CONFIDENCE_THRESHOLD is upgraded to "big"; an
unparseable reply routes "big").

It makes real model calls: free with the default local Gemma 4 E4B, billed if
SMALL_PROVIDER points at a paid model. No answers are generated.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from config import CONFIDENCE_THRESHOLD, ML_ROUTER_CONFIDENCE_THRESHOLD, ML_ROUTER_PATH
from llm.small_llm import SmallLLM
from llm.usage import track_usage
from modules.classifier import Classifier
from router.eval_data import load_cases
from router.ml_router import MLRouter

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "live_cascade_results.json")
# Per-query classifier verdicts from --all, so any router's cascade can be
# scored offline (router/refine.py) without re-asking the model.
VERDICTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "classifier_verdicts_1000.jsonl")


def final_route(clf) -> str:
    """The orchestrator's rule: only a confident "small" stays small; anything
    else (low confidence, "big", an unexpected value) is dispatched big."""
    if clf.routing_decision == "small" and clf.confidence >= CONFIDENCE_THRESHOLD:
        return "small"
    return "big"


def classify_all(classifier: Classifier, queries: list[str], label: str) -> list[dict]:
    out = []
    for i, q in enumerate(queries, 1):
        with track_usage() as ledger:
            clf = classifier.classify(q)
        out.append({
            "query": q,
            "llm_decision": clf.routing_decision,
            "llm_confidence": clf.confidence,
            "parsed": bool(clf.raw),
            "final": final_route(clf),
            "latency_ms": sum(c.latency_ms for c in ledger.calls),
            "input_tokens": ledger.input_tokens,
            "output_tokens": ledger.output_tokens,
        })
        if i % 25 == 0 or i == len(queries):
            print(f"  {label}: {i}/{len(queries)}", flush=True)
    return out


def summarize(rows: list[dict], truth: list[str]) -> dict:
    lat = [r["latency_ms"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": sum(r["final"] == t for r, t in zip(rows, truth)) / len(rows),
        "big_share": sum(r["final"] == "big" for r in rows) / len(rows),
        "parse_failures": sum(not r["parsed"] for r in rows),
        "upgraded_low_confidence": sum(
            r["llm_decision"] == "small" and r["final"] == "big" for r in rows),
        "latency_ms_median": statistics.median(lat),
        "latency_ms_p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--all", action="store_true",
                    help="also run the LLM classifier alone on all 1000 queries")
    args = ap.parse_args()

    small = SmallLLM()
    classifier = Classifier(small)
    cases = load_cases("eval_1000.jsonl")
    router = MLRouter.load(ML_ROUTER_PATH)
    preds = [router.predict(q) for q, _, _ in cases]
    deferred = [i for i, p in enumerate(preds) if p.confidence < ML_ROUTER_CONFIDENCE_THRESHOLD]

    print(f"\nLive cascade eval — classifier on {small.provider_name}/{small.model}")
    print(f"ML router defers on {len(deferred)}/{len(cases)} queries "
          f"(confidence < {ML_ROUTER_CONFIDENCE_THRESHOLD})\n")
    started = time.time()
    rows = classify_all(classifier, [cases[i][0] for i in deferred], "deferred")

    # Full cascade: ML decision where confident, classifier's final route elsewhere
    final = [p.decision for p in preds]
    for i, r in zip(deferred, rows):
        final[i] = r["final"]
    truth = [gt for _, gt, _ in cases]
    ml_only_on_deferred = sum(preds[i].decision == truth[i] for i in deferred) / len(deferred)
    deferred_summary = summarize(rows, [truth[i] for i in deferred])
    cascade = {
        "accuracy": sum(f == t for f, t in zip(final, truth)) / len(truth),
        "big_share": sum(f == "big" for f in final) / len(final),
    }

    print(f"\n{'':44}{'Accuracy':>10}{'Large %':>10}")
    print("─" * 64)
    print(f"{'Deferred queries: ML router guess':<44}{ml_only_on_deferred:>10.1%}")
    print(f"{'Deferred queries: LLM classifier (final)':<44}"
          f"{deferred_summary['accuracy']:>10.1%}{deferred_summary['big_share']:>10.1%}")
    print(f"{'Full cascade, all 1000':<44}{cascade['accuracy']:>10.1%}{cascade['big_share']:>10.1%}")
    print(f"\nClassifier replies that failed to parse: {deferred_summary['parse_failures']}"
          f"   'small' verdicts upgraded (conf < {CONFIDENCE_THRESHOLD}): "
          f"{deferred_summary['upgraded_low_confidence']}")
    print(f"Classifier latency: median {deferred_summary['latency_ms_median']:.0f} ms, "
          f"p95 {deferred_summary['latency_ms_p95']:.0f} ms")

    result = {
        "small_model": f"{small.provider_name}/{small.model}",
        "deferred": deferred_summary,
        "ml_guess_accuracy_on_deferred": ml_only_on_deferred,
        "cascade": cascade,
    }
    if args.all:
        print("\nLLM classifier alone on all 1000 queries:")
        all_rows = classify_all(classifier, [q for q, _, _ in cases], "all")
        result["classifier_only"] = summarize(all_rows, truth)
        with open(VERDICTS_PATH, "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps({k: r[k] for k in (
                    "query", "llm_decision", "llm_confidence", "parsed", "final")}) + "\n")
        s = result["classifier_only"]
        print(f"  accuracy {s['accuracy']:.1%}, large {s['big_share']:.1%}, "
              f"parse failures {s['parse_failures']}, median {s['latency_ms_median']:.0f} ms")

    if args.all:
        print(f"Per-query verdicts saved to {os.path.normpath(VERDICTS_PATH)}")
    print(f"\nWall time: {time.time() - started:.0f} s")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
