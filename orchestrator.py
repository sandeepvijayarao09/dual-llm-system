"""
Orchestration Engine — the central brain of the Dual LLM System.

Flow:
  1. Load user profile from DB (auto, by user_id)
  2. Pre-flight checks (token count)
  3. Classify with Small LLM -> routing decision (small | big)
  4. Route:
       "small" -> SimpleAnswerer (Small LLM, personalized via profile)
       "big"   -> Reasoner (Large LLM, personalized via profile injection)
                  -> ContextAdder adds a short personal note (query + profile only)
  5. After session: ProfileUpdater extracts signals -> ProfileDB merges updates

Key design decisions:
  - Large LLM output is NEVER passed to Small LLM (avoids large input problem).
  - Personalization for Large LLM is done via system prompt injection.
  - Profile evolves automatically after every session.
  - ProfileUpdater only sees user queries (never LLM answers).
"""

import logging
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

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    """Structured response returned by the orchestrator for every query."""
    answer: str
    routing_decision: str       # "small" | "big" | "pre-flight:large-input"
    model_used: str              # which model produced the main answer
    context_note: str            # personalised note added by Small LLM (may be empty)
    classification: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class Orchestrator:
    """Central orchestrator that routes queries between Small and Large LLMs."""

    def __init__(self) -> None:
        logger.info("Initialising Dual LLM System...")

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

        logger.info("Small LLM : %s (OpenAI)", self.small_llm.model)
        logger.info("Large LLM : %s (OpenAI)", self.large_llm.model)
        logger.info("System ready.")

    # ── Profile helpers ───────────────────────────────────────────────────────

    def load_profile(self, user_id: str) -> dict:
        """Load a user's profile from the database by user_id."""
        return self.profile_db.get(user_id)

    def end_session(
        self,
        user_id: str,
        session_queries: list[str],
    ) -> dict:
        """
        Call at the end of a session to dynamically update the user profile.

        Steps:
          1. ProfileUpdater analyzes session queries (Small LLM, small input)
          2. ProfileDB merges extracted signals into existing profile
          3. Returns the updated profile

        Only user queries are analyzed — never LLM answers.
        """
        if not session_queries:
            return self.profile_db.get(user_id)

        logger.info("Updating profile for user '%s' (%d queries)...", user_id, len(session_queries))

        current_profile = self.profile_db.get(user_id)
        updates = self.profile_updater.extract_updates(session_queries, current_profile)

        logger.debug("Extracted updates: %s", updates)

        updated_profile = self.profile_db.merge_updates(user_id, updates)

        logger.info("Profile updated. Interaction #%d", updated_profile["interaction_count"])
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
        Process a user query end-to-end and return the response with metadata.

        If user_id is provided, the profile is auto-loaded from the DB.
        A user_profile dict can still be passed directly to override DB lookup.
        """
        # Auto-load profile from DB if user_id given and no explicit profile
        if user_id and not user_profile:
            user_profile = self.profile_db.get(user_id)

        start = time.time()

        try:
            # ── Step 1: Pre-flight ─────────────────────────────────────────
            token_count = len(query.split())
            if token_count > LARGE_INPUT_THRESHOLD:
                logger.info("Pre-flight: %d words — routing directly to Large LLM", token_count)
                return self._route_big(
                    query, history, user_profile,
                    "pre-flight:large-input", start
                )

            # ── Step 2: Classify ───────────────────────────────────────────
            clf = self.classifier.classify(query)

            logger.info(
                "Classification: %s (complexity=%s, confidence=%.2f, intent=%s)",
                clf.routing_decision, clf.complexity, clf.confidence, clf.intent,
            )

            # Override to "big" if confidence too low for a "small" decision
            routing = clf.routing_decision
            if routing == "small" and clf.confidence < CONFIDENCE_THRESHOLD:
                routing = "big"
                logger.info(
                    "Confidence %.2f < %.2f — upgrading to Large LLM",
                    clf.confidence, CONFIDENCE_THRESHOLD,
                )

            # ── Step 3: Route ──────────────────────────────────────────────
            if routing == "small":
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
            logger.exception("Error processing query: %s", exc)
            return OrchestratorResponse(
                answer=f"Error: {exc}",
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
        history: list[dict] | None,
        user_profile: dict | None,
        routing: str,
        start: float,
        classification: dict | None = None,
    ) -> OrchestratorResponse:
        """
        Flow:
          1. Large LLM answers (NO profile — pure core concept)
          2. Small LLM summarizes the answer in 1 line
          3. Small LLM generates personalized intro + closing
             (using query + profile + 1-line summary, never full output)
          4. Final = intro + core answer + closing
        """
        # Step 1: Large LLM — core concept only, no profile
        core_answer = self.reasoner.reason(query, history)

        # Step 2: Small LLM — 1-line summary for context awareness
        answer_summary = self.context_adder.summarize(core_answer)
        logger.debug("Answer summary: %s", answer_summary)

        # Step 3: Small LLM — personalized intro + closing
        context = self.context_adder.add_context(query, user_profile, answer_summary)
        intro = context.get("intro", "")
        closing = context.get("closing", "")

        # Step 4: Assemble final response
        parts = []
        if intro:
            parts.append(intro)
            parts.append("")   # blank line
        parts.append(core_answer)
        if closing:
            parts.append("")
            parts.append("---")
            parts.append(f"💡 {closing}")

        final_answer = "\n".join(parts)

        return OrchestratorResponse(
            answer=final_answer,
            routing_decision=routing,
            model_used=self.large_llm.model,
            context_note=closing,
            classification=classification or {},
            latency_ms=self._ms(start),
        )

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.time() - start) * 1000, 1)
