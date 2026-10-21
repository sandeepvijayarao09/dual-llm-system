"""Tests for router.cost_eval — the offline routing-cost comparison."""

from router.cost_eval import CostModel, evaluate
from router.ml_router import RouterPrediction

CASES = [("hi", "small", "greetings"),
         ("Prove that the square root of two is irrational in full detail", "big", "proofs")]


def _pred(decision, confidence):
    return RouterPrediction(decision=decision, confidence=confidence, probs={})


def test_rows_are_relative_to_always_large():
    rows = {r["strategy"]: r for r in evaluate(
        CostModel(150, 600), CASES, [_pred("small", 0.9), _pred("big", 0.9)])}
    assert rows["Always Large LLM"]["vs_always_large"] == 1.0
    oracle = rows["Oracle (ground-truth labels)"]
    assert oracle["accuracy"] == 1.0
    assert oracle["cost_per_1k"] < rows["Always Large LLM"]["cost_per_1k"]


def test_cascaded_queries_pay_for_the_classifier():
    confident = evaluate(CostModel(150, 600), CASES, [_pred("small", 0.9), _pred("big", 0.9)])
    unsure = evaluate(CostModel(150, 600), CASES, [_pred("small", 0.5), _pred("big", 0.9)])
    assert unsure[3]["cost_per_1k"] > confident[3]["cost_per_1k"]
    # The bound row sends the cascaded "hi" to the Large LLM
    assert unsure[4]["large_share"] == 1.0
