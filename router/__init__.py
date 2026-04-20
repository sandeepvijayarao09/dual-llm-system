"""
Router package — ML-based query routing (small vs. large LLM).

Flow at inference time:
    query --> features.encode_text --> MLRouter.predict
                                       --> (routing_decision, confidence)

Fallback:
    If confidence < threshold, caller should defer to the LLM-based
    Classifier (modules/classifier.py). This is the RouteLLM cascade pattern.
"""

from router.ml_router import MLRouter, RouterPrediction
from router.classification_logger import ClassificationLogger

__all__ = ["MLRouter", "RouterPrediction", "ClassificationLogger"]
