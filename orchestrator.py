"""
Orchestration Engine — the central brain of the Dual LLM System.

Flow:
  1. Pre-flight checks (token count)
  2. Classify with Small LLM → routing decision (small | big)
  3. Route:
       "small" → SimpleAnswerer (Small LLM, personalized via profile)
                 → no context note needed, already personalized
       "big"   → Reasoner (Large LLM, personalized via profile injection)
                 → ContextAdder (Small LLM) adds a short personal note
                 → final output = Large answer + context note

Key design decisions:
  - Large LLM output is NEVER passed to Small LLM (avoids large input problem).
  - Personalization for Large LLM is done via system prompt injection (~50-80 tokens).
  - Small LLM only adds a 2-3 sentence context note (query + profile, always small).
"""

import time
from dataclasses import dataclass, field
from config import CONFIDENCE_THRESHOLD, LARGE_INPUT_THRESHOLD
from llm.small_llm import SmallLLM
from llm.large_llm import LargeLLM
from modules.classifier import Classifier
from modules.answerer import SimpleAnswerer
from modules.reasoner import Reasoner
from modules.personalizer import Personalizer   # acts as ContextAdder


@dataclass
class OrchestratorResponse:
    answer: str
    routing_decision: str       # "small" | "big" | "pre-flight:large-input"
    model_used: str              # which model produced the main answer
    context_note: str            # personalised note added by Small LLM (may be empty)
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
        self.classifier    = Classifier(self.small_llm)
        self.answerer      = SimpleAnswerer(self.small_llm)
        self.reasoner      = Reasoner(self.large_llm)
        self.context_adder = Personalizer(self.small_llm)   # ContextAdder

        print(f"  Small LLM : {self.small_llm.model} (Google Gemini)")
        print(f"  Large LLM : {self.large_llm.model} (Google Gemini)")
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
                return self._route_big(
                    query, history, user_profile,
                    "pre-flight:large-input", start
                )

            # ── Step 2: Classify ───────────────────────────────────────────
            if verbose:
                print("  → Classifying query with Small LLM...")

            clf = self.classifier.classify(query)

            if verbose:
                print(f"  → Classification: {clf.routing_decision} "
                      f"(complexity={clf.complexity}, confidence={clf.confidence:.2f}, "
                      f"intent={clf.intent})")

            # Override to "big" if confidence too low for a "small" decision
            routing = clf.routing_decision
            if routing == "small" and clf.confidence < CONFIDENCE_THRESHOLD:
                routing = "big"
                if verbose:
                    print(f"  → Confidence {clf.confidence:.2f} < {CONFIDENCE_THRESHOLD} "
                          "— upgrading to Large LLM")

            # ── Step 3: Route ──────────────────────────────────────────────
            if routing == "small":
                # Small LLM answers directly — personalized via profile
                if verbose:
                    print("  → Answering with Small LLM (personalized)...")
                answer = self.answerer.answer(query, history, user_profile)
                return OrchestratorResponse(
                    answer=answer,
                    routing_decision="small",
                    model_used=self.small_llm.model,
                    context_note="",
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
                context_note="",
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
        """
        Large LLM answers (personalized via profile injection).
        Small LLM adds a short context note (never sees the large output).
        """
        # Large LLM: answers + personalizes in one shot
        main_answer = self.reasoner.reason(
            query, history, user_profile=user_profile
        )

        # Small LLM: adds context note (query + profile only, never the main answer)
        context_note = self.context_adder.add_context(query, user_profile)

        # Assemble final response
        if context_note:
            final_answer = f"{main_answer}\n\n---\n💡 For you:\n{context_note}"
        else:
            final_answer = main_answer

        return OrchestratorResponse(
            answer=final_answer,
            routing_decision=routing,
            model_used=self.large_llm.model,
            context_note=context_note,
            classification=classification or {},
            latency_ms=self._ms(start),
        )

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.time() - start) * 1000, 1)
