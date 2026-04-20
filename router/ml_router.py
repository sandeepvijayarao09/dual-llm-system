"""
MLRouter — scikit-learn classifier for query routing.

Pipeline (serialized as a single joblib file):
    TfidfVectorizer(ngram_range=(1,2))  →  LogisticRegression

Inference:
    >>> router = MLRouter.load("router/models/router_v0.joblib")
    >>> pred = router.predict("Prove that sqrt 2 is irrational.")
    >>> pred.decision         # "big"
    >>> pred.confidence       # 0.91

Fallback contract:
    Caller should defer to the LLM-based classifier when
    `pred.confidence < confidence_threshold`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from router.features import encode_text

logger = logging.getLogger(__name__)


@dataclass
class RouterPrediction:
    decision: str              # "small" | "big"
    confidence: float          # prob of the predicted class, 0.0–1.0
    probs: dict                # {"small": ..., "big": ...}


class MLRouter:
    """Thin wrapper around a fitted sklearn Pipeline."""

    CLASSES = ("big", "small")   # keep order stable for serialization

    def __init__(self, pipeline: Optional[object] = None):
        self.pipeline = pipeline

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> "MLRouter":
        import joblib
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Router model not found at {path}. "
                "Run: python -m router.train"
            )
        pipeline = joblib.load(path)
        return cls(pipeline=pipeline)

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.pipeline, path)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, query: str) -> RouterPrediction:
        if self.pipeline is None:
            raise RuntimeError("MLRouter has no fitted pipeline loaded.")

        encoded = encode_text(query)
        proba = self.pipeline.predict_proba([encoded])[0]
        classes = list(self.pipeline.classes_)
        probs = {c: float(p) for c, p in zip(classes, proba)}

        decision = max(probs, key=probs.get)     # type: ignore[arg-type]
        confidence = probs[decision]

        return RouterPrediction(
            decision=decision,
            confidence=confidence,
            probs=probs,
        )
