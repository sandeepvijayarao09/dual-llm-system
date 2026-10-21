"""
cost_eval.py — Estimated serving cost of each routing strategy on Eval-1000.

Usage:
    python -m router.cost_eval [--small-out 150] [--big-out 600] [--no-profile]

No LLM calls are made. Each query is priced as the calls this system would
make for it, using the real system prompts and the list prices in
llm/usage.py:

    small route : answerer (Small LLM)
    big route   : reasoner (Large LLM) + summarizer + personalizer (Small LLM)
    cascade     : + LLM classifier (Small LLM) when ML confidence < 0.65

Assumptions (state them whenever you quote a number from this script):
    - tokens ≈ characters / 4 for prompts (no tokenizer dependency);
    - answer lengths are fixed (--small-out / --big-out) because the eval has
      no model outputs; the sensitivity table varies --big-out;
    - a cascaded query keeps the ML router's decision. Offline we do not have
      the LLM classifier's answer, so the "cascaded → big" row gives the bound
      where the classifier sends every cascaded query to the Large LLM;
    - misrouted queries are priced, not penalised: a big query answered by the
      Small LLM is cheaper and worse. Read cost next to accuracy.

Calibrate with real traffic: ClassificationLogger.cost_summary() reports
measured tokens and cost per route from routing_log.db.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from config import LARGE_MODEL, SMALL_MODEL, ML_ROUTER_PATH, ML_ROUTER_CONFIDENCE_THRESHOLD
from llm.usage import PRICES_AS_OF, estimate_cost
from modules.answerer import ANSWERER_SYSTEM
from modules.classifier import CLASSIFIER_SYSTEM
from modules.personalizer import CONTEXT_SYSTEM
from modules.reasoner import REASONER_SYSTEM
from router.ablation import baseline_length
from router.eval_data import load_cases

SUMMARIZER_SYSTEM = "Summarize the following text in exactly one sentence. Output only the summary."
CLASSIFIER_OUT = 60      # JSON verdict
SUMMARY_OUT = 30         # one sentence
PERSONALIZER_OUT = 60    # INTRO + CLOSING lines
PROFILE_TOKENS = 150     # a preset profile serialised as JSON
WORD_RULE_MAX = 8        # FINDINGS.md §3: ≤8 words → small (same \w+ count as ablation)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "cost_results_1000.json")


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class CostModel:
    small_out: int
    big_out: int
    with_profile: bool = True

    def small_route(self, q: str) -> float:
        return estimate_cost(SMALL_MODEL, tokens(ANSWERER_SYSTEM) + tokens(q), self.small_out)

    def big_route(self, q: str) -> float:
        cost = estimate_cost(LARGE_MODEL, tokens(REASONER_SYSTEM) + tokens(q), self.big_out)
        cost += estimate_cost(SMALL_MODEL, tokens(SUMMARIZER_SYSTEM) + self.big_out, SUMMARY_OUT)
        if self.with_profile:
            cost += estimate_cost(
                SMALL_MODEL,
                tokens(CONTEXT_SYSTEM) + PROFILE_TOKENS + tokens(q) + SUMMARY_OUT,
                PERSONALIZER_OUT,
            )
        return cost

    def classifier(self, q: str) -> float:
        return estimate_cost(SMALL_MODEL, tokens(CLASSIFIER_SYSTEM) + tokens(q), CLASSIFIER_OUT)

    def route(self, q: str, decision: str) -> float:
        return self.big_route(q) if decision == "big" else self.small_route(q)


def evaluate(cm: CostModel, cases, preds) -> list[dict]:
    """One row per strategy: accuracy, Large-LLM share, cost per 1k queries."""
    n = len(cases)
    cascaded = [p.confidence < ML_ROUTER_CONFIDENCE_THRESHOLD for p in preds]

    def row(name, decisions, extra_cost=0.0):
        cost = sum(cm.route(q, d) for (q, _, _), d in zip(cases, decisions)) + extra_cost
        acc = sum(d == gt for (_, gt, _), d in zip(cases, decisions)) / n
        return {"strategy": name, "accuracy": acc,
                "large_share": sum(d == "big" for d in decisions) / n,
                "cost_per_1k": cost / n * 1000}

    classifier_cost = sum(cm.classifier(q) for (q, _, _), c in zip(cases, cascaded) if c)
    ml = [p.decision for p in preds]
    rows = [
        row("Always Large LLM", ["big"] * n),
        row("Always Small LLM", ["small"] * n),
        row(f"Word-count rule (<={WORD_RULE_MAX} words -> small)",
            [baseline_length(q, WORD_RULE_MAX) for q, _, _ in cases]),
        row("Cascade router (ML decision)", ml, classifier_cost),
        row("Cascade router (cascaded -> big, bound)",
            ["big" if c else d for d, c in zip(ml, cascaded)], classifier_cost),
        row("Oracle (ground-truth labels)", [gt for _, gt, _ in cases]),
    ]
    base = rows[0]["cost_per_1k"]
    for r in rows:
        r["vs_always_large"] = r["cost_per_1k"] / base
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--small-out", type=int, default=150, help="Small-LLM answer tokens")
    ap.add_argument("--big-out", type=int, default=600, help="Large-LLM answer tokens")
    ap.add_argument("--no-profile", action="store_true", help="guest user: no personalizer call")
    args = ap.parse_args()

    if estimate_cost(LARGE_MODEL, 1, 1) is None or estimate_cost(SMALL_MODEL, 1, 1) is None:
        raise SystemExit(f"No price for {LARGE_MODEL} or {SMALL_MODEL}: "
                         "add it to MODEL_PRICES or MODEL_PRICES_PATH.")

    from router.ml_router import MLRouter
    cases = load_cases("eval_1000.jsonl")
    router = MLRouter.load(ML_ROUTER_PATH)
    preds = [router.predict(q) for q, _, _ in cases]

    cm = CostModel(args.small_out, args.big_out, with_profile=not args.no_profile)
    rows = evaluate(cm, cases, preds)

    print(f"\nEstimated serving cost on Eval-1000 — {SMALL_MODEL} / {LARGE_MODEL}, "
          f"list prices as of {PRICES_AS_OF}")
    print(f"Answer length: small {args.small_out} tok, big {args.big_out} tok; "
          f"personalizer {'on' if cm.with_profile else 'off'}\n")
    print(f"{'Strategy':<44}{'Accuracy':>9}{'Large %':>9}{'$ / 1k q':>10}{'vs Large':>10}")
    print("─" * 82)
    for r in rows:
        print(f"{r['strategy']:<44}{r['accuracy']:>9.1%}{r['large_share']:>9.1%}"
              f"{r['cost_per_1k']:>10.3f}{r['vs_always_large']:>10.1%}")

    print("\nSensitivity — cascade router cost as a share of Always Large:")
    sensitivity = {}
    for big_out in (300, 600, 1200):
        s_rows = evaluate(CostModel(args.small_out, big_out, cm.with_profile), cases, preds)
        share = s_rows[3]["vs_always_large"]
        sensitivity[big_out] = share
        print(f"  big answer {big_out:>5} tok: {share:.1%}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"prices_as_of": PRICES_AS_OF, "small_model": SMALL_MODEL,
                   "large_model": LARGE_MODEL, "assumptions": vars(args),
                   "rows": rows, "cascade_share_by_big_out": sensitivity}, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
