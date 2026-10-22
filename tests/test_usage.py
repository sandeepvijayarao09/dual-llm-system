"""Tests for llm.usage — price lookup, cost estimates and the per-query ledger."""

import threading
import time

import pytest

from llm.usage import estimate_cost, price_for, record_call, stage, track_usage


def test_cost_uses_per_million_list_prices():
    # gemini-3.8-flash: $0.75 in / $3.75 out per 1M tokens
    assert estimate_cost("gemini-3.8-flash", 1_000_000, 0) == pytest.approx(0.75)
    assert estimate_cost("gemini-3.8-flash", 0, 1_000_000) == pytest.approx(3.75)
    assert estimate_cost("gemini-4-argon", 1000, 500) == pytest.approx(
        (1000 * 2.00 + 500 * 10.00) / 1e6)


def test_cached_input_is_billed_at_the_cached_rate():
    full = estimate_cost("gemini-3.8-flash", 1000, 0)
    half_cached = estimate_cost("gemini-3.8-flash", 1000, 0, cached_input_tokens=500)
    # half the tokens at $0.075 instead of $0.75
    assert half_cached == pytest.approx(full * 0.55)


def test_versioned_names_resolve_to_their_base_model():
    assert price_for("gemini-3.8-flash-001") == price_for("gemini-3.8-flash")
    # Ollama tags resolve on ":" and are free (local inference)
    assert price_for("gemma4:e4b") == price_for("gemma4")
    assert estimate_cost("gemma4:e4b-it-q4_K_M", 10_000, 10_000) == 0.0
    assert estimate_cost("gemma-4-31b-it", 10_000, 10_000) == 0.0


def test_unpriced_model_gives_none_not_a_guess():
    assert estimate_cost("some-new-model", 100, 100) is None
    with track_usage() as ledger:
        record_call("gemini-3.8-flash", time.perf_counter(), 10, 10)
        record_call("some-new-model", time.perf_counter(), 10, 10)
    assert ledger.cost_usd is None            # never a partial total
    assert ledger.input_tokens == 20


def test_calls_are_labelled_by_stage_and_ignored_outside_tracking():
    record_call("gemini-3.8-flash", time.perf_counter(), 1, 1)     # no active ledger: no-op
    with track_usage() as ledger:
        with stage("classifier"):
            record_call("gemma4:e4b", time.perf_counter(), 5, 2)
        record_call("gemini-3.8-flash", time.perf_counter(), 5, 2)
    assert [c.stage for c in ledger.calls] == ["classifier", "unlabelled"]


def test_failed_call_is_recorded_without_cost():
    with track_usage() as ledger:
        record_call("gemini-3.8-flash", time.perf_counter(), error="RateLimitError")
    assert ledger.calls[0].error == "RateLimitError"
    assert ledger.calls[0].cost_usd is None


def test_ledgers_are_isolated_between_threads():
    """The API shares one Orchestrator across request threads."""
    barrier = threading.Barrier(2)
    seen = {}

    def worker(name, n):
        with track_usage() as ledger:
            barrier.wait()
            for _ in range(n):
                record_call("gemini-3.8-flash", time.perf_counter(), 1, 1)
            seen[name] = len(ledger.calls)

    threads = [threading.Thread(target=worker, args=("a", 3)),
               threading.Thread(target=worker, args=("b", 5))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"a": 3, "b": 5}
