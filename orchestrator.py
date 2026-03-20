"""
Orchestration Engine — the central brain of the Dual LLM System.

Flow:
  1. Pre-flight checks (token count, cache lookup)
  2. Classify with Small LLM  →  routing decision
  3. Route:
       "small"          → SimpleAnswerer  (Small LLM)
       "big"            → Reasoner        (Large LLM)
       "big_then_small" → Reasoner → Personalizer
  4. Return response + metadata
"""

import time
from dataclasses import dataclass, field
from config import CONFIDENCE_THRESHOLD, LARGE_INPUT_THRESHOLD
from llm.small_llm import SmallLLM
from llm.large_llm import LargeLLM
from modules.classifier import Classifier
from modules.answerer import SimpleAnswerer
from modules.reasoner import Reasoner
from modules.personalizer import Personalizer


@dataclass
class OrchestratorResponse:
    answer: str
    routing_decision: str          # "small" | "big" | "big_then_small"
    model_used: str                 # which model produced the final answer
    personalizer_applied: bool
    classification: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class Orchestrator:
    def __init__(self):
        print("🔧 Initialising Dual LLM System...")

        # Initialise LLM clients
        self.small_llm = SmallLLM()
        self.large_llm = LargeLLM()

        # Initialise modules
        self.classifier = Classifier(self.small_llm)
        self.answerer = SimpleAnswerer(self.small_llm)
        self.reasoner = Reasoner(self.large_llm)
        self.personalizer = Personalizer(self.small_llm)

        print(f"  Small LLM : {self.small_llm.model} (Ollama local)")
        print(f"  Large LLM : {self.large_llm.model} (Anthropic API)")
        print("✅ Ready\n")

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(
        self,
        query: str,
        history: list[dict] | None = None,
        user_profile: dict | None = None,
        verbose: bool = False,
    ) -> OrchestratorResponse:
        """
        Process a user query end-to-end and return the response with metadata.
        """
        start = time.time()

        try:
            # ── Step 1: Pre-flight ─────────────────────────────────────────
            token_count = len(query.split())
            if token_count > LARGE_INPUT_THRESHOLD:
                if verbose:
                    print(f"  → Pre-flight: {token_count} tokens — routing directly to Large LLM")
                return self._route_big(query, history, user_profile, "pre-flight:large-input", start)

            # ── Step 2: Classify ───────────────────────────────────────────
            if verbose:
                print("  → Classifying query with Small LLM...")

            clf = self.classifier.classify(query)

            if verbose:
                print(f"  → Classification: {clf.routing_decision} "
                      f"(complexity={clf.complexity}, confidence={clf.confidence:.2f}, "
                      f"intent={clf.intent})")

            # Override to "big" if confidence is too low for a "small" decision
            routing = clf.routing_decision
            if routing == "small" and clf.confidence < CONFIDENCE_THRESHOLD:
                routing = "big"
                if verbose:
                    print(f"  → Confidence {clf.confidence:.2f} < {CONFIDENCE_THRESHOLD} "
                          "— upgrading to Large LLM")

            # ── Step 3: Route ──────────────────────────────────────────────
            if routing == "small":
                answer = self.answerer.answer(query, history, user_profile)
                return OrchestratorResponse(
                    answer=answer,
                    routing_decision="small",
                    model_used=self.small_llm.model,
                    personalizer_applied=False,
                    classification=clf.raw,
                    latency_ms=self._ms(start),
                )

            elif routing == "big_then_small":
                if verbose:
                    print("  → Reasoning with Large LLM...")
                raw = self.reasoner.reason(query, history)
                if verbose:
                    print("  → Personalising with Small LLM...")
                answer = self.personalizer.personalize(raw, user_profile, query)
                return OrchestratorResponse(
                    answer=answer,
                    routing_decision="big_then_small",
                    model_used=self.large_llm.model,
                    personalizer_applied=True,
                    classification=clf.raw,
                    latency_ms=self._ms(start),
                )

            else:  # "big"
                return self._route_big(query, history, user_profile, routing, start, clf.raw)

        except Exception as exc:
            return OrchestratorResponse(
                answer=f"❌ Error: {exc}",
                routing_decision="error",
                model_used="none",
                personalizer_applied=False,
                latency_ms=self._ms(start),
                error=str(exc),
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _route_big(
        self,
        query: str,
        history,
        user_profile,
        routing: str,
        start: float,
        classification: dict | None = None,
    ) -> OrchestratorResponse:
        answer = self.reasoner.reason(query, history)
        return OrchestratorResponse(
            answer=answer,
            routing_decision=routing,
            model_used=self.large_llm.model,
            personalizer_applied=False,
            classification=classification or {},
            latency_ms=self._ms(start),
        )

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.time() - start) * 1000, 1)
