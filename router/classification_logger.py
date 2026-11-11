"""
ClassificationLogger — append-only log of every routing decision.

Purpose:
    Capture real-world (query, ml_prediction, llm_prediction, final_decision,
    model_used) tuples during production so we can periodically retrain the
    MLRouter on *your users' actual traffic*, not just the seed set.

Schema: `routing_events` (one row per query, with summed tokens, estimated
cost and latency) and `llm_calls` (one row per API request), defined in
db/engine.py. Stored in ROUTING_LOG_PATH (SQLite) or DATABASE_URL.

Exporting training data:
    logger.export_labeled_csv("data.csv")
      → rows where ml_decision and llm_decision AGREE become high-confidence
        training labels (agreement is a proxy for correctness)

Cost reporting:
    logger.cost_summary()
      → totals and per-route breakdown of tokens, estimated $ and latency
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import UTC, datetime

from sqlalchemy import func, insert, select
from sqlalchemy.exc import SQLAlchemyError

from db.engine import database_url, get_engine, llm_calls, routing_events

logger = logging.getLogger(__name__)

_CALL_FIELDS = ("stage", "model", "input_tokens", "output_tokens",
                "cached_input_tokens", "latency_ms", "cost_usd", "error")


class ClassificationLogger:
    def __init__(self, db_path: str = "routing_log.db"):
        """`db_path` is a SQLite file path or any SQLAlchemy URL."""
        self.db_path = db_path
        self.engine = get_engine(database_url(db_path))

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(
        self,
        query: str,
        final_routing: str,
        user_id: str | None = None,
        ml_decision: str | None = None,
        ml_confidence: float | None = None,
        llm_decision: str | None = None,
        llm_confidence: float | None = None,
        model_used: str | None = None,
        routed_by: str | None = None,
        latency_ms: float | None = None,
        usage: dict | None = None,
    ) -> None:
        """`usage` is UsageLedger.summary(); its calls go to llm_calls."""
        usage = usage or {}
        try:
            with self.engine.begin() as conn:
                result = conn.execute(
                    insert(routing_events).values(
                        ts=datetime.now(UTC).isoformat(),
                        user_id=user_id,
                        query=query,
                        ml_decision=ml_decision,
                        ml_confidence=ml_confidence,
                        llm_decision=llm_decision,
                        llm_confidence=llm_confidence,
                        final_routing=final_routing,
                        model_used=model_used,
                        routed_by=routed_by,
                        latency_ms=latency_ms,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        cached_input_tokens=usage.get("cached_input_tokens"),
                        cost_usd=usage.get("cost_usd"),
                    )
                )
                event_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
                calls = usage.get("calls", [])
                if calls:
                    conn.execute(insert(llm_calls), [
                        {"event_id": event_id, **{k: c.get(k) for k in _CALL_FIELDS}}
                        for c in calls
                    ])
        except SQLAlchemyError as exc:                  # pragma: no cover
            logger.warning("Routing log write failed: %s", exc)

    # ── Read / export ────────────────────────────────────────────────────────

    def export_labeled_csv(self, out_path: str, agreement_only: bool = True) -> int:
        """
        Write a (query,label) CSV suitable for `python -m router.train --extra`.

        If agreement_only=True (default), only emit rows where the ML router
        and LLM classifier agreed — those are our highest-quality labels.
        Returns number of rows written.
        """
        t = routing_events.c
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(t.query, t.ml_decision, t.llm_decision, t.final_routing)
            ).mappings().all()

        kept: list[tuple[str, str]] = []
        for row in rows:
            q = row["query"]
            ml, llm, final = row["ml_decision"], row["llm_decision"], row["final_routing"]
            if agreement_only:
                if ml and llm and ml == llm:
                    kept.append((q, ml))
            else:
                if final in {"small", "big"}:
                    kept.append((q, final))

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "label"])
            writer.writerows(kept)
        return len(kept)

    def cost_summary(self) -> dict:
        """
        Totals plus a per-route breakdown over every logged query:
            {"queries", "cost_usd", "unpriced", "input_tokens", "output_tokens",
             "avg_latency_ms", "by_route": {route: {...same keys...}}}
        cost_usd sums only priced rows; "unpriced" counts the rest.
        """
        t = routing_events.c
        cols = [
            func.count().label("queries"),
            func.sum(t.cost_usd).label("cost_usd"),
            (func.count() - func.count(t.cost_usd)).label("unpriced"),
            func.coalesce(func.sum(t.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(t.output_tokens), 0).label("output_tokens"),
            func.avg(t.latency_ms).label("avg_latency_ms"),
        ]
        with self.engine.connect() as conn:
            total = dict(conn.execute(select(*cols)).mappings().one())
            rows = conn.execute(
                select(t.final_routing, *cols).group_by(t.final_routing)
            ).mappings().all()
        total["by_route"] = {
            r["final_routing"]: {k: v for k, v in r.items() if k != "final_routing"}
            for r in rows
        }
        return total

    def calls(self) -> list[dict]:
        """Every recorded LLM call, oldest first."""
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(llm_calls)).mappings()]

    def count(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(routing_events)).scalar_one()
