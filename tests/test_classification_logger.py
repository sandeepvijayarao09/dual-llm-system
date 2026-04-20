"""Tests for router.classification_logger.ClassificationLogger."""

import csv
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
