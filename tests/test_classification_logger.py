"""Tests for router.classification_logger.ClassificationLogger."""

import csv

import pytest

from router.classification_logger import ClassificationLogger


def test_log_and_count(tmp_path):
    log = ClassificationLogger(str(tmp_path / "r.db"))
    assert log.count() == 0

    log.log(
        query="What is HTTP?",
        final_routing="small",
        user_id="u1",
        ml_decision="small", ml_confidence=0.82,
        llm_decision="small", llm_confidence=0.9,
        model_used="fake-small",
    )
    assert log.count() == 1


def test_export_agreement_only(tmp_path):
    log = ClassificationLogger(str(tmp_path / "r.db"))

    # Agreement → should be exported
    log.log(query="hi", final_routing="small",
            ml_decision="small", llm_decision="small")
    # Disagreement → should be dropped when agreement_only=True
    log.log(query="ambiguous?", final_routing="big",
            ml_decision="small", llm_decision="big")
    # No ML decision → dropped under agreement_only
    log.log(query="pure-llm?", final_routing="big",
            ml_decision=None, llm_decision="big")

    out = tmp_path / "train.csv"
    n = log.export_labeled_csv(str(out), agreement_only=True)
    assert n == 1

    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"query": "hi", "label": "small"}]


def test_export_all_final_decisions(tmp_path):
    log = ClassificationLogger(str(tmp_path / "r.db"))
    log.log(query="q1", final_routing="small")
    log.log(query="q2", final_routing="big")
    log.log(query="q3", final_routing="error")   # should be skipped

    out = tmp_path / "all.csv"
    n = log.export_labeled_csv(str(out), agreement_only=False)
    assert n == 2


def test_usage_is_logged_per_query_and_per_call(tmp_path):
    log = ClassificationLogger(str(tmp_path / "r.db"))
    usage = {
        "input_tokens": 300, "output_tokens": 80, "cached_input_tokens": 0,
        "cost_usd": 0.0012,
        "calls": [
            {"stage": "reasoner", "model": "gpt-4o", "input_tokens": 200,
             "output_tokens": 60, "cached_input_tokens": 0, "latency_ms": 900.0,
             "cost_usd": 0.0011, "error": None},
            {"stage": "personalizer", "model": "gpt-4o-mini", "input_tokens": 100,
             "output_tokens": 20, "cached_input_tokens": 0, "latency_ms": 300.0,
             "cost_usd": 0.0001, "error": None},
        ],
    }
    log.log(query="Prove it", final_routing="big", routed_by="ml",
            latency_ms=1250.0, usage=usage)
    log.log(query="hi", final_routing="small", routed_by="ml",
            latency_ms=200.0, usage={"input_tokens": 10, "output_tokens": 2,
                                     "cost_usd": None, "calls": []})

    stages = [c["stage"] for c in log.calls()]
    assert stages == ["reasoner", "personalizer"]

    summary = log.cost_summary()
    assert summary["queries"] == 2
    assert summary["cost_usd"] == pytest.approx(0.0012)
    assert summary["unpriced"] == 1
    assert summary["by_route"]["big"]["input_tokens"] == 300


def test_old_database_is_migrated(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE routing_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, user_id TEXT, query TEXT NOT NULL, ml_decision TEXT, "
            "ml_confidence REAL, llm_decision TEXT, llm_confidence REAL, "
            "final_routing TEXT NOT NULL, model_used TEXT)"
        )
        conn.execute("INSERT INTO routing_events (ts, query, final_routing) "
                     "VALUES ('t', 'old row', 'small')")

    log = ClassificationLogger(path)
    log.log(query="new row", final_routing="small", usage={"cost_usd": 0.5})
    assert log.count() == 2
    assert log.cost_summary()["cost_usd"] == pytest.approx(0.5)
