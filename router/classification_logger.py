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
        model_used     TEXT
    );

Exporting training data:
    logger.export_labeled_csv("data.csv")
      → rows where ml_decision and llm_decision AGREE become high-confidence
        training labels (agreement is a proxy for correctness)
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


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
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO routing_events
                    (ts, user_id, query, ml_decision, ml_confidence,
                     llm_decision, llm_confidence, final_routing, model_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
