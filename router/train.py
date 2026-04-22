"""
Train the v0 MLRouter on the seed dataset.

Usage:
    python -m router.train                    # train + save + print metrics
    python -m router.train --out path.joblib  # custom output path
    python -m router.train --extra data.csv   # merge extra CSV (query,label)

Extra CSV schema:
    query,label
    "What is HTTP?",small
    "Design a URL shortener",big
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from router.features import encode_text
from router.ml_router import MLRouter
from router.seed_data import load_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUT = os.path.join("router", "models", "router_v0.joblib")


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),       # add trigrams
            min_df=1,
            max_features=5000,        # larger vocab
            sublinear_tf=True,
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            C=2.0,                    # slightly less regularization
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",           # better for larger feature sets
        )),
    ])


def _load_extra(path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q, lbl = row.get("query"), row.get("label")
            if q and lbl in {"small", "big"}:
                rows.append((q.strip(), lbl))
    return rows


def train(
    dataset: Iterable[tuple[str, str]],
    out_path: str = DEFAULT_OUT,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    data = list(dataset)
    if not data:
        raise ValueError("Empty dataset.")

    X = [encode_text(q) for q, _ in data]
    y = [lbl for _, lbl in data]

    # Stratified train/test split for honest metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    logger.info("Training done on %d examples (test=%d).", len(X_train), len(X_test))
    logger.info("Test accuracy: %.3f", acc)
    logger.info("\n%s", report)

    # Refit on all data before saving — more data = better production model
    pipeline.fit(X, y)
    MLRouter(pipeline=pipeline).save(out_path)
    logger.info("Saved model → %s", out_path)

    return {
        "test_accuracy": acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": list(pipeline.classes_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v0 ML router.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output .joblib path")
    parser.add_argument("--extra", default=None, help="Extra CSV data to merge")
    args = parser.parse_args()

    data = load_seed()
    if args.extra:
        extras = _load_extra(args.extra)
        logger.info("Merging %d extra rows from %s", len(extras), args.extra)
        data.extend(extras)

    metrics = train(data, out_path=args.out)
    print(f"\nFinal: accuracy={metrics['test_accuracy']:.3f} "
          f"on {metrics['n_test']} test examples "
          f"(classes={metrics['classes']})")


if __name__ == "__main__":
    main()
