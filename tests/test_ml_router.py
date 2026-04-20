"""
Tests for router.ml_router.MLRouter + the training pipeline.

These tests train a small model from the seed dataset into a tmp_path and
then exercise predict/save/load. They require sklearn+joblib but no network.
"""

import os

from router.ml_router import MLRouter
from router.seed_data import load_seed
from router.train import train


def test_train_produces_usable_model(tmp_path):
    out = tmp_path / "model.joblib"
    metrics = train(load_seed(), out_path=str(out))

    assert out.exists()
    # Our seed set should train well above random (50/50)
    assert metrics["test_accuracy"] > 0.75
    assert set(metrics["classes"]) == {"small", "big"}


def test_load_and_predict(tmp_path):
    out = tmp_path / "model.joblib"
    train(load_seed(), out_path=str(out))

    r = MLRouter.load(str(out))
    pred = r.predict("Hi there!")
    assert pred.decision in {"small", "big"}
    assert 0.0 <= pred.confidence <= 1.0
    assert set(pred.probs.keys()) == {"small", "big"}
    # Probs sum to ~1
    total = sum(pred.probs.values())
    assert abs(total - 1.0) < 1e-6


def test_routes_obvious_big_queries(tmp_path):
    out = tmp_path / "model.joblib"
    train(load_seed(), out_path=str(out))
    r = MLRouter.load(str(out))

    big_queries = [
        "Prove that there are infinitely many prime numbers.",
        "Design a distributed rate limiter across data centers.",
        "Compare Rust Go and Zig for writing a database engine.",
        "Derive the closed-form solution for linear regression.",
    ]
    hits = sum(r.predict(q).decision == "big" for q in big_queries)
    # Allow one miss — seed is small; we just want the bias to be right
    assert hits >= len(big_queries) - 1


def test_routes_obvious_small_queries(tmp_path):
    out = tmp_path / "model.joblib"
    train(load_seed(), out_path=str(out))
    r = MLRouter.load(str(out))

    small_queries = [
        "Hi!",
        "What is the capital of Japan?",
        "Thanks!",
        "What is 2 + 2?",
    ]
    hits = sum(r.predict(q).decision == "small" for q in small_queries)
    assert hits >= len(small_queries) - 1


def test_load_missing_file_raises(tmp_path):
    import pytest
    missing = tmp_path / "nope.joblib"
    with pytest.raises(FileNotFoundError):
        MLRouter.load(str(missing))


def test_predict_requires_fitted_pipeline():
    import pytest
    r = MLRouter(pipeline=None)
    with pytest.raises(RuntimeError):
        r.predict("anything")


def test_existing_v0_model_on_disk_loads():
    """If the committed v0 model exists, it must be loadable."""
    path = os.path.join("router", "models", "router_v0.joblib")
    if not os.path.exists(path):
        import pytest
        pytest.skip("v0 model not present — run `python -m router.train` first")

    r = MLRouter.load(path)
    pred = r.predict("Prove sqrt 2 is irrational.")
    assert pred.decision in {"small", "big"}
