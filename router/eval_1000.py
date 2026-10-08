"""
eval_1000.py — Offline evaluation of the ML Router against 1000 labeled test cases.

Usage:
    python -m router.eval_1000

Requirements:
    - router/models/router_v0.joblib must exist  (run: python -m router.train)
    - scikit-learn, joblib installed

No LLM calls are made.  Only the MLRouter (TF-IDF + Logistic Regression) is
evaluated.  Cascade rate is computed as a simulation: queries where
ML confidence < ML_ROUTER_CONFIDENCE_THRESHOLD (0.65) *would* fall back to
the LLM classifier in production.

Key differences from eval_500.py:
    - 1000 entirely new queries (no overlap with eval_500.py)
    - Four new categories: conversational, debug_simple,
      ethical_philosophical, comparison_technical
    - Before/After comparison table (eval_500 results are hardcoded)
    - Results saved to router/eval_results_1000.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass, asdict
from typing import Optional

from router.eval_data import load_cases

# ---------------------------------------------------------------------------
# 1000 labeled test cases — ALL NEW, none reused from eval_500.py
# Stored in data/eval_1000.jsonl; loaded as (query, ground_truth_label, category)
# ground_truth: "small" | "big"
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, str]] = load_cases("eval_1000.jsonl")

# Verify count
assert len(TEST_CASES) == 1000, f"Expected 1000 test cases, got {len(TEST_CASES)}"


# ---------------------------------------------------------------------------
# Hardcoded eval_500 baseline results (from the 500-case evaluation)
# Used to build the Before/After comparison table.
# ---------------------------------------------------------------------------
EVAL_500_BASELINE: dict[str, dict] = {
    "_overall": {"accuracy": 0.882, "cascade_rate": 0.308},
    "ambiguous":       {"accuracy": 0.367, "cascade_rate": 0.467, "n": 30},
    "short_creative":  {"accuracy": 0.900, "cascade_rate": 1.000, "n": 20},
    "career_learning": {"accuracy": 0.950, "cascade_rate": 0.800, "n": 20},
    "research_synthesis": {"accuracy": 0.850, "cascade_rate": 0.600, "n": 20},
    "greetings":       {"accuracy": 1.000, "cascade_rate": 0.000, "n": 20},
    "factual":         {"accuracy": 0.917, "cascade_rate": 0.033, "n": 60},
    "math_simple":     {"accuracy": 1.000, "cascade_rate": 0.000, "n": 30},
    "yes_no":          {"accuracy": 1.000, "cascade_rate": 0.050, "n": 20},
    "definitions":     {"accuracy": 0.900, "cascade_rate": 0.133, "n": 30},
    "trivia":          {"accuracy": 0.850, "cascade_rate": 0.100, "n": 20},
    "conversions":     {"accuracy": 1.000, "cascade_rate": 0.000, "n": 20},
    "proofs":          {"accuracy": 0.950, "cascade_rate": 0.050, "n": 20},
    "system_design":   {"accuracy": 0.967, "cascade_rate": 0.067, "n": 30},
    "multi_step_code": {"accuracy": 0.933, "cascade_rate": 0.033, "n": 30},
    "analysis":        {"accuracy": 0.933, "cascade_rate": 0.067, "n": 30},
    "compound_questions": {"accuracy": 0.967, "cascade_rate": 0.033, "n": 30},
    "real_world_planning": {"accuracy": 0.950, "cascade_rate": 0.050, "n": 20},
    "multi_constraint":{"accuracy": 0.950, "cascade_rate": 0.050, "n": 20},
    "domain_specific_deep": {"accuracy": 0.933, "cascade_rate": 0.033, "n": 30},
}

# ---------------------------------------------------------------------------
# Evaluation dataclass
# ---------------------------------------------------------------------------

CASCADE_THRESHOLD = 0.65  # matches config.ML_ROUTER_CONFIDENCE_THRESHOLD


@dataclass
class QueryResult:
    query: str
    ground_truth: str
    category: str
    ml_decision: str
    ml_confidence: float
    correct: bool
    would_cascade: bool


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(model_path: str) -> list[QueryResult]:
    """Load the ML router and evaluate all 1000 test cases."""
    from router.ml_router import MLRouter

    print(f"\nLoading ML Router from: {model_path}")
    router = MLRouter.load(model_path)
    print("Model loaded successfully.\n")

    results: list[QueryResult] = []
    for query, ground_truth, category in TEST_CASES:
        pred = router.predict(query)
        correct = pred.decision == ground_truth
        would_cascade = pred.confidence < CASCADE_THRESHOLD

        results.append(QueryResult(
            query=query,
            ground_truth=ground_truth,
            category=category,
            ml_decision=pred.decision,
            ml_confidence=pred.confidence,
            correct=correct,
            would_cascade=would_cascade,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _header(title: str, width: int = 76) -> str:
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}"


def _subheader(title: str, width: int = 76) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def _delta(new_val: float, old_val: Optional[float], pct: bool = True) -> str:
    """Format a delta string with + / - sign."""
    if old_val is None:
        return "  (new)"
    diff = new_val - old_val
    sign = "+" if diff >= 0 else ""
    if pct:
        return f"  ({sign}{diff:.1%})"
    return f"  ({sign}{diff:.3f})"


def print_report(results: list[QueryResult]) -> None:  # noqa: C901
    total = len(results)
    correct_all = sum(1 for r in results if r.correct)
    overall_accuracy = correct_all / total

    # Confusion matrix
    tp = sum(1 for r in results if r.ground_truth == "big" and r.ml_decision == "big")
    fp = sum(1 for r in results if r.ground_truth == "small" and r.ml_decision == "big")
    tn = sum(1 for r in results if r.ground_truth == "small" and r.ml_decision == "small")
    fn = sum(1 for r in results if r.ground_truth == "big" and r.ml_decision == "small")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Confidence distribution
    confidences = [r.ml_confidence for r in results]
    conf_mean   = statistics.mean(confidences)
    conf_median = statistics.median(confidences)
    conf_min    = min(confidences)
    conf_max    = max(confidences)
    above_thresh = sum(1 for c in confidences if c >= CASCADE_THRESHOLD)

    # Cascade stats
    cascade_count = sum(1 for r in results if r.would_cascade)
    cascade_rate  = cascade_count / total

    # Per-category breakdown
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    # ── Print report ─────────────────────────────────────────────────────────

    print(_header("ML ROUTER EVALUATION REPORT — 1000 LABELED TEST CASES"))
    print(f"\n  Total test cases : {total}")
    print(f"  Cascade threshold: {CASCADE_THRESHOLD}")
    print(f"  New categories   : conversational, debug_simple, ethical_philosophical, comparison_technical")

    # ── 1. Overall accuracy ──────────────────────────────────────────────────
    print(_subheader("1. OVERALL ML ROUTER ACCURACY"))
    baseline_acc = EVAL_500_BASELINE["_overall"]["accuracy"]
    print(f"  Correct predictions : {correct_all} / {total}")
    print(f"  Accuracy (1000-case): {overall_accuracy:.1%}")
    print(f"  Accuracy (500-case baseline): {baseline_acc:.1%}  →  delta: {_delta(overall_accuracy, baseline_acc).strip()}")

    # ── 2. Confusion matrix ──────────────────────────────────────────────────
    print(_subheader("2. CONFUSION MATRIX  (positive class = 'big')"))
    print(f"  {'':30s} {'Predicted BIG':>14} {'Predicted SMALL':>15}")
    print(f"  {'Actual BIG':30s} {'TP =':>8} {tp:>5}   {'FN =':>8} {fn:>5}")
    print(f"  {'Actual SMALL':30s} {'FP =':>8} {fp:>5}   {'TN =':>8} {tn:>5}")
    print()
    print(f"  Precision (big class) : {precision:.3f}")
    print(f"  Recall    (big class) : {recall:.3f}")
    print(f"  F1 score  (big class) : {f1:.3f}")

    # ── 3. Confidence distribution ───────────────────────────────────────────
    print(_subheader("3. CONFIDENCE DISTRIBUTION"))
    print(f"  Mean   : {conf_mean:.3f}")
    print(f"  Median : {conf_median:.3f}")
    print(f"  Min    : {conf_min:.3f}")
    print(f"  Max    : {conf_max:.3f}")
    print(f"  Above cascade threshold ({CASCADE_THRESHOLD}): {above_thresh} / {total}  "
          f"({above_thresh/total:.1%})")

    # Histogram
    print()
    print("  Confidence histogram:")
    bucket_width = 0.10
    for i in range(10):
        lo = i * bucket_width
        hi = lo + bucket_width
        if i == 9:
            count = sum(1 for c in confidences if lo <= c <= 1.0)
        else:
            count = sum(1 for c in confidences if lo <= c < hi)
        bar = "#" * min(count, 60)
        print(f"    [{lo:.1f} – {hi:.1f}) : {count:>4}  {bar}")

    # ── 4. Cascade rate ───────────────────────────────────────────────────────
    print(_subheader("4. CASCADE RATE  (ML confidence < threshold → LLM fallback)"))
    baseline_cascade = EVAL_500_BASELINE["_overall"]["cascade_rate"]
    print(f"  Would cascade       : {cascade_count} / {total}  ({cascade_rate:.1%})")
    print(f"  Handled by ML alone : {total - cascade_count} / {total}  ({(total-cascade_count)/total:.1%})")
    print(f"  500-case baseline cascade rate: {baseline_cascade:.1%}  →  delta: {_delta(cascade_rate, baseline_cascade).strip()}")

    # ── 5. Per-category accuracy & cascade rate ───────────────────────────────
    print(_subheader("5. PER-CATEGORY BREAKDOWN"))
    cat_names = sorted(categories.keys())
    col1 = max(len(n) for n in cat_names) + 2
    header_line = (
        f"  {'Category':<{col1}} {'N':>4}  {'Acc':>7}  {'Correct':>7}  "
        f"{'CascadeRate':>11}  {'Cascades':>8}  {'Label'}"
    )
    print(header_line)
    print("  " + "-" * (len(header_line) - 2))

    for cat in cat_names:
        cat_results = categories[cat]
        n = len(cat_results)
        n_correct = sum(1 for r in cat_results if r.correct)
        n_cascade = sum(1 for r in cat_results if r.would_cascade)
        acc = n_correct / n
        casc_rate = n_cascade / n
        label = "(NEW)" if cat in {"conversational", "debug_simple", "ethical_philosophical", "comparison_technical"} else ""
        print(
            f"  {cat:<{col1}} {n:>4}  {acc:>7.1%}  {n_correct:>7}  "
            f"{casc_rate:>11.1%}  {n_cascade:>8}  {label}"
        )

    # ── 6. BEFORE / AFTER COMPARISON TABLE ───────────────────────────────────
    print(_subheader("6. BEFORE / AFTER IMPROVEMENT TABLE  (500-case baseline vs 1000-case)"))
    shared_cats = [c for c in cat_names if c in EVAL_500_BASELINE]
    col1b = max(len(c) for c in shared_cats) + 2

    hdr = (f"  {'Category':<{col1b}} "
           f"{'Acc-500':>8} {'Acc-1000':>9} {'AccDelta':>9}  "
           f"{'Casc-500':>9} {'Casc-1000':>10} {'CascDelta':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cat in shared_cats:
        cat_results = categories[cat]
        n = len(cat_results)
        n_correct = sum(1 for r in cat_results if r.correct)
        n_cascade = sum(1 for r in cat_results if r.would_cascade)
        acc_1000 = n_correct / n
        casc_1000 = n_cascade / n

        b = EVAL_500_BASELINE[cat]
        acc_500 = b["accuracy"]
        casc_500 = b["cascade_rate"]

        acc_delta = acc_1000 - acc_500
        casc_delta = casc_1000 - casc_500

        acc_d_str = f"{'+' if acc_delta >= 0 else ''}{acc_delta:.1%}"
        casc_d_str = f"{'+' if casc_delta >= 0 else ''}{casc_delta:.1%}"

        print(
            f"  {cat:<{col1b}} "
            f"{acc_500:>8.1%} {acc_1000:>9.1%} {acc_d_str:>9}  "
            f"{casc_500:>9.1%} {casc_1000:>10.1%} {casc_d_str:>10}"
        )

    # Overall row
    oa_delta = overall_accuracy - EVAL_500_BASELINE["_overall"]["accuracy"]
    casc_d_oa = cascade_rate - EVAL_500_BASELINE["_overall"]["cascade_rate"]
    oa_d_str = f"{'+' if oa_delta >= 0 else ''}{oa_delta:.1%}"
    casc_d_oa_str = f"{'+' if casc_d_oa >= 0 else ''}{casc_d_oa:.1%}"
    print("  " + "-" * (len(hdr) - 2))
    print(
        f"  {'OVERALL':<{col1b}} "
        f"{EVAL_500_BASELINE['_overall']['accuracy']:>8.1%} {overall_accuracy:>9.1%} {oa_d_str:>9}  "
        f"{EVAL_500_BASELINE['_overall']['cascade_rate']:>9.1%} {cascade_rate:>10.1%} {casc_d_oa_str:>10}"
    )

    # ── 7. Top 20 wrong predictions ───────────────────────────────────────────
    print(_subheader("7. TOP 20 WRONG PREDICTIONS  (highest ML confidence)"))
    wrong = [r for r in results if not r.correct]
    wrong_sorted = sorted(wrong, key=lambda r: r.ml_confidence, reverse=True)[:20]

    if not wrong_sorted:
        print("  No wrong predictions! Perfect accuracy.")
    else:
        for i, r in enumerate(wrong_sorted, 1):
            q_display = r.query[:70] + ("..." if len(r.query) > 70 else "")
            print(
                f"  {i:>2}. [{r.category}] conf={r.ml_confidence:.3f} "
                f"gt={r.ground_truth} pred={r.ml_decision}\n"
                f"      \"{q_display}\""
            )

    # ── 8. Top 20 most uncertain predictions ──────────────────────────────────
    print(_subheader("8. TOP 20 MOST UNCERTAIN PREDICTIONS  (lowest confidence)"))
    uncertain = sorted(results, key=lambda r: r.ml_confidence)[:20]

    for i, r in enumerate(uncertain, 1):
        status = "CORRECT" if r.correct else "WRONG"
        q_display = r.query[:70] + ("..." if len(r.query) > 70 else "")
        print(
            f"  {i:>2}. [{r.category}] conf={r.ml_confidence:.3f} "
            f"gt={r.ground_truth} pred={r.ml_decision}  [{status}]\n"
            f"      \"{q_display}\""
        )

    # ── 9. New-category summary ────────────────────────────────────────────────
    print(_subheader("9. NEW CATEGORY RESULTS  (not present in eval_500.py)"))
    new_cats = ["conversational", "debug_simple", "ethical_philosophical", "comparison_technical"]
    for cat in new_cats:
        if cat not in categories:
            print(f"  {cat}: (no cases)")
            continue
        cr = categories[cat]
        n = len(cr)
        acc = sum(1 for r in cr if r.correct) / n
        casc = sum(1 for r in cr if r.would_cascade) / n
        print(f"  {cat:<30} N={n:>4}  Accuracy={acc:.1%}  CascadeRate={casc:.1%}")

    # ── 10. Summary recommendation ─────────────────────────────────────────────
    print(_subheader("10. SUMMARY & RECOMMENDATION"))

    issues: list[str] = []
    positives: list[str] = []

    if overall_accuracy >= 0.90:
        positives.append(f"Strong overall accuracy ({overall_accuracy:.1%}).")
    elif overall_accuracy >= 0.80:
        positives.append(f"Acceptable overall accuracy ({overall_accuracy:.1%}); room for improvement.")
    else:
        issues.append(f"Low overall accuracy ({overall_accuracy:.1%}) — retrain with more data.")

    if cascade_rate <= 0.15:
        positives.append(f"Low cascade rate ({cascade_rate:.1%}) — ML router handles most queries confidently.")
    elif cascade_rate <= 0.30:
        issues.append(
            f"Moderate cascade rate ({cascade_rate:.1%}) — "
            f"roughly {cascade_count} queries would fall back to the LLM classifier."
        )
    else:
        issues.append(
            f"High cascade rate ({cascade_rate:.1%}) — {cascade_count} queries require LLM fallback."
        )

    if precision < 0.85:
        issues.append(
            f"Precision for 'big' class is {precision:.1%} — too many small queries misrouted to large LLM."
        )
    if recall < 0.85:
        issues.append(
            f"Recall for 'big' class is {recall:.1%} — too many complex queries handled by small LLM."
        )

    cat_accs = {
        cat: sum(1 for r in cr if r.correct) / len(cr)
        for cat, cr in categories.items()
    }
    worst_cats = sorted(cat_accs, key=cat_accs.get)[:3]  # type: ignore[arg-type]
    bad_str = ", ".join(
        f"{c} ({cat_accs[c]:.0%})" for c in worst_cats if cat_accs[c] < 0.80
    )
    if bad_str:
        issues.append(f"Weakest categories: {bad_str}. Add more seed examples for these.")

    # Compare to baseline
    if overall_accuracy > EVAL_500_BASELINE["_overall"]["accuracy"]:
        positives.append(
            f"Accuracy improved over 500-case baseline "
            f"({EVAL_500_BASELINE['_overall']['accuracy']:.1%} → {overall_accuracy:.1%})."
        )
    if cascade_rate < EVAL_500_BASELINE["_overall"]["cascade_rate"]:
        positives.append(
            f"Cascade rate improved over 500-case baseline "
            f"({EVAL_500_BASELINE['_overall']['cascade_rate']:.1%} → {cascade_rate:.1%})."
        )

    print()
    if positives:
        print("  Strengths:")
        for p in positives:
            print(f"    + {p}")
    if issues:
        print("\n  Issues / Action items:")
        for item in issues:
            print(f"    ! {item}")

    print("\n  Recommended next steps:")
    if overall_accuracy < 0.90:
        print("    1. Add 50-100 more labeled examples for weak categories to router/seed_data.py")
        print("    2. Retrain with: python -m router.train")
    if cascade_rate > 0.20:
        print("    3. Consider tuning ML_ROUTER_CONFIDENCE_THRESHOLD (currently 0.65).")
    print("    4. Collect real production queries via ClassificationLogger and retrain periodically.")

    print("\n" + "=" * 76 + "\n")


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

def save_results(results: list[QueryResult], out_path: str) -> None:
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    total = len(results)
    correct_all = sum(1 for r in results if r.correct)
    cascade_count = sum(1 for r in results if r.would_cascade)

    per_cat: dict = {}
    for cat, cr in categories.items():
        n = len(cr)
        per_cat[cat] = {
            "n": n,
            "accuracy": sum(1 for r in cr if r.correct) / n,
            "cascade_rate": sum(1 for r in cr if r.would_cascade) / n,
        }

    output = {
        "total_cases": total,
        "overall_accuracy": correct_all / total,
        "overall_cascade_rate": cascade_count / total,
        "cascade_threshold": CASCADE_THRESHOLD,
        "per_category": per_cat,
        "baseline_500": EVAL_500_BASELINE,
        "results": [asdict(r) for r in results],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "router", "models", "router_v0.joblib")
    out_path = os.path.join(project_root, "router", "eval_results_1000.json")

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Train the model first with:  python -m router.train")
        sys.exit(1)

    results = run_evaluation(model_path)
    print_report(results)
    save_results(results, out_path)


if __name__ == "__main__":
    main()
