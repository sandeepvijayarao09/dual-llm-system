"""
eval_500.py — Offline evaluation of the ML Router against 500 labeled test cases.

Usage:
    python -m router.eval_500

Requirements:
    - router/models/router_v0.joblib must exist  (run: python -m router.train)
    - scikit-learn, joblib installed

No LLM calls are made.  Only the MLRouter (TF-IDF + Logistic Regression) is
evaluated.  Cascade rate is computed as a simulation: queries where
ML confidence < ML_ROUTER_CONFIDENCE_THRESHOLD (0.65) *would* fall back to
the LLM classifier in production.
"""

from __future__ import annotations

import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

from router.eval_data import load_cases

# ---------------------------------------------------------------------------
# 500 labeled test cases
# Stored in data/eval_500.jsonl; loaded as (query, ground_truth_label, category)
# ground_truth: "small" | "big"
# category: human-readable bucket used in per-category breakdown
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, str]] = load_cases("eval_500.jsonl")

# Verify count
assert len(TEST_CASES) == 500, f"Expected 500 test cases, got {len(TEST_CASES)}"


# ---------------------------------------------------------------------------
# Evaluation dataclass
# ---------------------------------------------------------------------------

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

CASCADE_THRESHOLD = 0.65  # matches config.ML_ROUTER_CONFIDENCE_THRESHOLD


def run_evaluation(model_path: str) -> list[QueryResult]:
    """Load the ML router and evaluate all 500 test cases."""
    from router.ml_router import MLRouter

    print(f"\nLoading ML Router from: {model_path}")
    router = MLRouter.load(model_path)
    print(f"Model loaded successfully.\n")

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

def _header(title: str, width: int = 72) -> str:
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}"


def _subheader(title: str, width: int = 72) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def print_report(results: list[QueryResult]) -> None:  # noqa: C901
    total = len(results)
    correct_all = sum(1 for r in results if r.correct)
    overall_accuracy = correct_all / total

    # Confusion matrix counts (positive class = "big")
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

    print(_header("ML ROUTER EVALUATION REPORT — 500 LABELED TEST CASES"))

    print(f"\n  Total test cases : {total}")
    print(f"  Cascade threshold: {CASCADE_THRESHOLD}")

    # ── 1. Overall accuracy ──────────────────────────────────────────────────
    print(_subheader("1. OVERALL ML ROUTER ACCURACY"))
    print(f"  Correct predictions : {correct_all} / {total}")
    print(f"  Accuracy            : {overall_accuracy:.1%}")

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

    # Histogram (10 buckets)
    print()
    bucket_width = 0.10
    print("  Confidence histogram:")
    for i in range(10):
        lo = i * bucket_width
        hi = lo + bucket_width
        count = sum(1 for c in confidences if lo <= c < hi)
        bar = "#" * count
        # Last bucket is inclusive
        if i == 9:
            count = sum(1 for c in confidences if lo <= c <= hi)
        print(f"    [{lo:.1f} – {hi:.1f}) : {count:>4}  {bar[:60]}")

    # ── 4. Cascade rate ───────────────────────────────────────────────────────
    print(_subheader("4. CASCADE RATE  (ML confidence < threshold → LLM fallback)"))
    print(f"  Would cascade       : {cascade_count} / {total}  ({cascade_rate:.1%})")
    print(f"  Handled by ML alone : {total - cascade_count} / {total}  "
          f"({(total - cascade_count)/total:.1%})")

    # ── 5. Per-category accuracy & cascade rate ───────────────────────────────
    print(_subheader("5. PER-CATEGORY BREAKDOWN"))
    cat_names = sorted(categories.keys())
    col1 = max(len(n) for n in cat_names) + 2
    header_line = (
        f"  {'Category':<{col1}} {'N':>4}  {'Acc':>7}  {'Correct':>7}  "
        f"{'CascadeRate':>11}  {'Cascades':>8}"
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
        print(
            f"  {cat:<{col1}} {n:>4}  {acc:>7.1%}  {n_correct:>7}  "
            f"{casc_rate:>11.1%}  {n_cascade:>8}"
        )

    # ── 6. Top 20 wrong predictions ───────────────────────────────────────────
    print(_subheader("6. TOP 20 WRONG PREDICTIONS  (highest ML confidence)"))
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

    # ── 7. Top 20 low-confidence predictions ─────────────────────────────────
    print(_subheader("7. TOP 20 MOST UNCERTAIN PREDICTIONS  (lowest confidence)"))
    uncertain = sorted(results, key=lambda r: r.ml_confidence)[:20]

    for i, r in enumerate(uncertain, 1):
        status = "CORRECT" if r.correct else "WRONG"
        q_display = r.query[:70] + ("..." if len(r.query) > 70 else "")
        print(
            f"  {i:>2}. [{r.category}] conf={r.ml_confidence:.3f} "
            f"gt={r.ground_truth} pred={r.ml_decision}  [{status}]\n"
            f"      \"{q_display}\""
        )

    # ── 8. Summary recommendation ─────────────────────────────────────────────
    print(_subheader("8. SUMMARY & RECOMMENDATION"))

    issues: list[str] = []
    positives: list[str] = []

    if overall_accuracy >= 0.90:
        positives.append(f"Strong overall accuracy ({overall_accuracy:.1%}).")
    elif overall_accuracy >= 0.80:
        positives.append(f"Acceptable overall accuracy ({overall_accuracy:.1%}); room for improvement.")
    else:
        issues.append(f"Low overall accuracy ({overall_accuracy:.1%}) — consider retraining with more data.")

    if cascade_rate <= 0.15:
        positives.append(f"Low cascade rate ({cascade_rate:.1%}) — ML router handles most queries confidently.")
    elif cascade_rate <= 0.30:
        issues.append(
            f"Moderate cascade rate ({cascade_rate:.1%}) — roughly {cascade_count} queries would "
            f"fall back to the LLM classifier."
        )
    else:
        issues.append(
            f"High cascade rate ({cascade_rate:.1%}) — {cascade_count} queries require LLM fallback. "
            "The ML router lacks confidence on many inputs; collect more training data."
        )

    if precision < 0.85:
        issues.append(
            f"Precision for 'big' class is {precision:.1%} — too many small queries are being "
            "misrouted to the large LLM (expensive false positives)."
        )
    if recall < 0.85:
        issues.append(
            f"Recall for 'big' class is {recall:.1%} — too many complex queries are being "
            "incorrectly handled by the small LLM (quality-degrading false negatives)."
        )

    # Identify worst categories
    cat_accs = {
        cat: sum(1 for r in cat_results if r.correct) / len(cat_results)
        for cat, cat_results in categories.items()
    }
    worst_cats = sorted(cat_accs, key=cat_accs.get)[:3]  # type: ignore[arg-type]
    if cat_accs[worst_cats[0]] < 0.80:
        bad_str = ", ".join(f"{c} ({cat_accs[c]:.0%})" for c in worst_cats if cat_accs[c] < 0.80)
        if bad_str:
            issues.append(f"Weakest categories: {bad_str}. Add more examples for these to seed_data.py.")

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
        print("       Lowering it reduces LLM calls but risks more ML errors.")
    if f1 < 0.85:
        print("    4. Review feature engineering in router/features.py for ambiguous categories.")
    print("    5. Collect real production queries via ClassificationLogger and retrain periodically.")

    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Resolve model path relative to project root (two levels up from router/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "router", "models", "router_v0.joblib")

    # Allow override via CLI arg
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Train the model first with:  python -m router.train")
        sys.exit(1)

    results = run_evaluation(model_path)
    print_report(results)


if __name__ == "__main__":
    main()
