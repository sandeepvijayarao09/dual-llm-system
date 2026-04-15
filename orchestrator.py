"""
Orchestration Engine — the central brain of the Dual LLM System.

Flow:
  1. Load user profile from DB (auto, by user_id)
  2. Pre-flight checks (token count)
  3. Classify with Small LLM → routing decision (small | big)
  4. Route:
       "small" → SimpleAnswerer (Small LLM, personalized via profile)
       "big"   → Reasoner (Large LLM, personalized via profile injection)
                 → ContextAdder adds a short personal note (query + profile only)
  5. After session: ProfileUpdater extracts signals → ProfileDB merges updates

Key design decisions:
  - Large LLM output is NEVER passed to Small LLM (avoids large input problem).
  - Personalization for Large LLM is done via system prompt injection (~50-80 tokens).
  - Profile evolves automatically after every session — no manual updates needed.
  - ProfileUpdater only sees user queries (never LLM answers) — always small input.
"""

import time
from dataclasses import dataclass, field
from config import CONFIDENCE_THRESHOLD, LARGE_INPUT_THRESHOLD
from llm.small_llm import SmallLLM
from llm.large_llm import LargeLLM
from modules.classifier import Classifier
from modules.answerer import SimpleAnswerer
from modules.reasoner import Reasoner
from modules.personalizer import Personalizer       # ContextAdder
from modules.profile_updater import ProfileUpdater
from db.profile_db import ProfileDB


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
        self.classifier      = Classifier(self.small_llm)
        self.answerer        = SimpleAnswerer(self.small_llm)
        self.reasoner        = Reasoner(self.large_llm)
        self.context_adder   = Personalizer(self.small_llm)
        self.profile_updater = ProfileUpdater(self.small_llm)
        self.profile_db      = ProfileDB()

        print(f"  Small LLM : {self.small_llm.model} (OpenAI)")
        print(f"  Large LLM : {self.large_llm.model} (OpenAI)")
        print("✅ Ready\n")

    # ── Profile helpers ───────────────────────────────────────────────────────

    def load_profile(self, user_id: str) -> dict:
        """Load a user's profile from the database by user_id."""
        return self.profile_db.get(user_id)

    def end_session(
        self,
        user_id: str,
        session_queries: list[str],
        verbose: bool = False,
    ) -> dict:
        """
        Call this at the end of a session to dynamically update the user profile.

        Steps:
          1. ProfileUpdater analyzes session queries (Small LLM, small input)
          2. ProfileDB merges extracted signals into existing profile
          3. Returns the updated profile

        Only user queries are analyzed — never LLM answers.
        """
        if not session_queries:
            return self.profile_db.get(user_id)

        if verbose:
            print(f"\n  → Updating profile for user '{user_id}'...")

        current_profile = self.profile_db.get(user_id)
        updates = self.profile_updater.extract_updates(session_queries, current_profile)

        if verbose:
            print(f"  → Extracted updates: {updates}")

        updated_profile = self.profile_db.merge_updates(user_id, updates)

        if verbose:
            print(f"  → Profile updated. Interaction #{updated_profile['interaction_count']}")

        return updated_profile

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(
        self,
        query: str,
        history: list[dict] | None = None,
        user_profile: dict | None = None,
        user_id: str | None = None,
        verbose: bool = False,
    ) -> OrchestratorResponse:
        """
        If user_id is provided, profile is auto-loaded from DB.
        user_profile dict can still be passed directly to override DB lookup.
        """
        # Auto-load profile from DB if user_id given and no explicit profile
        if user_id and not user_profile:
            user_profile = self.profile_db.get(user_id)
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
