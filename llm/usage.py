"""
Per-query token, latency and cost accounting.

Every LLM client call is recorded as an `LLMCall` (one per API request — the
same granularity as an OpenTelemetry GenAI span) into the ledger of the
query currently being processed. The ledger is held in a ContextVar, so a
single Orchestrator shared by many Streamlit sessions/threads never mixes up
two users' usage.

    with track_usage() as ledger:
        with stage("classifier"):
            small_llm.complete(...)
    ledger.summary()   # {"input_tokens": ..., "cost_usd": ..., "calls": [...]}

Cost is an *estimate* from the list prices in `MODEL_PRICES` (USD per 1M
tokens). A model missing from the table gets cost_usd=None rather than a
guess; override or extend the table with MODEL_PRICES_PATH (a JSON file of
{"model": {"input": .., "cached_input": .., "output": ..}}).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# ── Price table ──────────────────────────────────────────────────────────────
# USD per 1M tokens, provider list prices. Check https://openai.com/api/pricing
# and https://claude.com/pricing before relying on the numbers, and bump
# PRICES_AS_OF when you do. PRICES_AS_OF is the oldest check: the OpenAI rows
# come from 2026-07 listings, the Claude rows from Anthropic's 2026-10 table.
# Gemini models are not listed: add the ones you use through MODEL_PRICES_PATH
# (until then they log cost_usd = NULL).
PRICES_AS_OF = "2026-07"
MODEL_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o":            {"input": 2.50, "cached_input": 1.25,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "claude-opus-5-5":   {"input": 4.00, "cached_input": 0.20,  "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "cached_input": 0.20,  "output": 10.00},
    # Haiku 5.5 prompts over 100K tokens bill at $0.50 / $2.50; not modelled here
    "claude-haiku-5-5":  {"input": 0.10, "output": 0.50},
}


def _load_price_overrides() -> None:
    path = os.getenv("MODEL_PRICES_PATH")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as f:
            MODEL_PRICES.update(json.load(f))
    except (OSError, ValueError) as exc:
        logger.warning("Could not load MODEL_PRICES_PATH=%s (%s)", path, exc)


_load_price_overrides()


def price_for(model: str) -> dict[str, float] | None:
    """Look up a model's prices; dated snapshots ("gpt-4o-2024-08-06") fall
    back to their base name. Longest matching name wins, so gpt-4o-mini-*
    never resolves to gpt-4o."""
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    matches = [m for m in MODEL_PRICES if model.startswith(m + "-")]
    return MODEL_PRICES[max(matches, key=len)] if matches else None


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """Estimated USD for one call, or None when the model has no price."""
    p = price_for(model)
    if p is None:
        return None
    uncached = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached * p["input"]
        + cached_input_tokens * p.get("cached_input", p["input"])
        + output_tokens * p["output"]
    ) / 1_000_000


# ── Ledger ───────────────────────────────────────────────────────────────────

@dataclass
class LLMCall:
    stage: str                  # "classifier" | "answerer" | "reasoner" | ...
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    error: str | None = None  # exception class name when the call failed


@dataclass
class UsageLedger:
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def cached_input_tokens(self) -> int:
        return sum(c.cached_input_tokens for c in self.calls)

    @property
    def cost_usd(self) -> float | None:
        """Total estimated cost; None if any call's model is unpriced, so a
        partial sum is never shown as the full cost."""
        costs = [c.cost_usd for c in self.calls]
        if any(c is None for c in costs):
            return None
        return sum(costs)  # type: ignore[arg-type]

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost_usd": self.cost_usd,
            "llm_latency_ms": round(sum(c.latency_ms for c in self.calls), 1),
            "n_calls": len(self.calls),
            "prices_as_of": PRICES_AS_OF,
            "calls": [asdict(c) for c in self.calls],
        }


_ledger: ContextVar[UsageLedger | None] = ContextVar("usage_ledger", default=None)
_stage: ContextVar[str] = ContextVar("usage_stage", default="unlabelled")


@contextmanager
def track_usage() -> Iterator[UsageLedger]:
    """Collect every LLM call made inside the block into a fresh ledger."""
    ledger = UsageLedger()
    token = _ledger.set(ledger)
    try:
        yield ledger
    finally:
        _ledger.reset(token)


def current_ledger() -> UsageLedger | None:
    """The ledger of the query being processed, if any."""
    return _ledger.get()


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Label the LLM calls made inside the block (e.g. "classifier")."""
    token = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(token)


def record_call(
    model: str,
    started: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    error: str | None = None,
) -> None:
    """Append one call to the active ledger (no-op outside track_usage).
    `started` is the time.perf_counter() value taken before the request."""
    ledger = _ledger.get()
    if ledger is None:
        return
    ledger.calls.append(LLMCall(
        stage=_stage.get(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        cost_usd=None if error else estimate_cost(
            model, input_tokens, output_tokens, cached_input_tokens),
        error=error,
    ))
