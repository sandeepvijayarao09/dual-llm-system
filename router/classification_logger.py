"""
ClassificationLogger — append-only SQLite log of every routing decision.

Purpose:
    Capture real-world (query, ml_prediction, llm_prediction, final_decision,
    model_used) tuples during production so we can periodically retrain the
    MLRouter on *your users' actual traffic*, not just the seed set.

Schema (single table):
    CREATE TABLE routing_events (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        ts             TEXT NOT NULL,
        user_id        TEXT,
        query          TEXT NOT NULL,
        ml_decision    TEXT,       -- "small" | "big" | NULL
        ml_confidence  REAL,
        llm_decision   TEXT,       -- "small" | "big" | NULL
        llm_confidence REAL,
        final_routing  TEXT NOT NULL,
        model_used     TEXT,
        routed_by      TEXT,       -- "ml" | "llm" | "pre-flight" | "slash"
        latency_ms     REAL,       -- end-to-end, as seen by the user
        input_tokens   INTEGER,    -- summed over every LLM call for the query
        output_tokens  INTEGER,
        cached_input_tokens INTEGER,
        cost_usd       REAL        -- estimate; NULL if a model is unpriced
    );

    CREATE TABLE llm_calls (       -- one row per API request
        event_id       INTEGER REFERENCES routing_events(id),
        stage          TEXT,       -- "classifier" | "answerer" | "reasoner" | ...
        model          TEXT,
        input_tokens   INTEGER,
        output_tokens  INTEGER,
        cached_input_tokens INTEGER,
        latency_ms     REAL,
        cost_usd       REAL,
        error          TEXT
    );

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
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Columns added after the first release; _init_db adds any that an older
# routing_log.db is missing.
_USAGE_COLUMNS = {
    "routed_by": "TEXT",
    "latency_ms": "REAL",
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cached_input_tokens": "INTEGER",
    "cost_usd": "REAL",
}


class ClassificationLogger:
    def __init__(self, db_path: str = "routing_log.db"):
        self.db_path = db_path
        self._init_db()

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(
        self,
        query: str,
        final_routing: str,
        user_id: Optional[str] = None,
        ml_decision: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        llm_decision: Optional[str] = None,
        llm_confidence: Optional[float] = None,
        model_used: Optional[str] = None,
        routed_by: Optional[str] = None,
        latency_ms: Optional[float] = None,
        usage: Optional[dict] = None,
    ) -> None:
        """`usage` is UsageLedger.summary(); its calls go to llm_calls."""
        usage = usage or {}
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO routing_events
                    (ts, user_id, query, ml_decision, ml_confidence,
                     llm_decision, llm_confidence, final_routing, model_used,
                     routed_by, latency_ms, input_tokens, output_tokens,
                     cached_input_tokens, cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        user_id,
                        query,
                        ml_decision,
                        ml_confidence,
                        llm_decision,
                        llm_confidence,
                        final_routing,
                        model_used,
                        routed_by,
                        latency_ms,
                        usage.get("input_tokens"),
                        usage.get("output_tokens"),
                        usage.get("cached_input_tokens"),
                        usage.get("cost_usd"),
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO llm_calls
                    (event_id, stage, model, input_tokens, output_tokens,
                     cached_input_tokens, latency_ms, cost_usd, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (cur.lastrowid, c["stage"], c["model"], c["input_tokens"],
                         c["output_tokens"], c["cached_input_tokens"],
                         c["latency_ms"], c["cost_usd"], c["error"])
                        for c in usage.get("calls", [])
                    ],
                )
        except sqlite3.Error as exc:                    # pragma: no cover
            logger.warning("Routing log write failed: %s", exc)

    # ── Read / export ────────────────────────────────────────────────────────

    def export_labeled_csv(self, out_path: str, agreement_only: bool = True) -> int:
        """
        Write a (query,label) CSV suitable for `python -m router.train --extra`.

        If agreement_only=True (default), only emit rows where the ML router
        and LLM classifier agreed — those are our highest-quality labels.
        Returns number of rows written.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT query, ml_decision, llm_decision, final_routing
                FROM routing_events
                """
            )
            rows = cursor.fetchall()

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
            {"queries", "cost_usd", "input_tokens", "output_tokens",
             "avg_latency_ms", "by_route": {route: {...same keys...}}}
        cost_usd sums only priced rows; "unpriced" counts the rest.
        """
        select = """
            COUNT(*)                          AS queries,
            SUM(cost_usd)                     AS cost_usd,
            SUM(cost_usd IS NULL)             AS unpriced,
            COALESCE(SUM(input_tokens), 0)    AS input_tokens,
            COALESCE(SUM(output_tokens), 0)   AS output_tokens,
            AVG(latency_ms)                   AS avg_latency_ms
        """
        with self._connect() as conn:
            total = dict(conn.execute(f"SELECT {select} FROM routing_events").fetchone())
            rows = conn.execute(
                f"SELECT final_routing, {select} FROM routing_events GROUP BY final_routing"
            ).fetchall()
        total["by_route"] = {
            r["final_routing"]: {k: r[k] for k in r.keys() if k != "final_routing"}
            for r in rows
        }
        return total

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM routing_events").fetchone()
        return row[0] if row else 0

    # ── Internals ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_events (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts             TEXT NOT NULL,
                    user_id        TEXT,
                    query          TEXT NOT NULL,
                    ml_decision    TEXT,
                    ml_confidence  REAL,
                    llm_decision   TEXT,
                    llm_confidence REAL,
                    final_routing  TEXT NOT NULL,
                    model_used     TEXT
                )
                """
            )
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(routing_events)")}
            for col, typ in _USAGE_COLUMNS.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE routing_events ADD COLUMN {col} {typ}")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_calls (
                    event_id            INTEGER REFERENCES routing_events(id),
                    stage               TEXT,
                    model               TEXT,
                    input_tokens        INTEGER,
                    output_tokens       INTEGER,
                    cached_input_tokens INTEGER,
                    latency_ms          REAL,
                    cost_usd            REAL,
                    error               TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
