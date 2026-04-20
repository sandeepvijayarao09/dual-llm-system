"""
Orchestration Engine — the central brain of the Dual LLM System.

Flow:
  1. Load user profile from DB (auto, by user_id)
  2. Pre-flight check (token count)
  3. Route:
       a. Try ML router first (fast, offline) if enabled
       b. If ML confidence < threshold → fall back to LLM classifier
       c. If ML confidence ≥ threshold → use its decision directly
  4. Dispatch:
       "small" → SimpleAnswerer (Small LLM, personalized via profile)
       "big"   → Reasoner (Large LLM, core answer)
                 → Personalizer adds intro/closing (query + profile + 1-line summary)
  5. Record the turn in a per-user ConversationBuffer (sliding window)
  6. Log the decision for future router training (ClassificationLogger)
  7. After session: ProfileUpdater extracts signals → ProfileDB merges updates

Design principles:
  - Large LLM output is NEVER passed to Small LLM in full (cost/context safety).
  - Context size is bounded per-user via ConversationBuffer (window + summary).
  - ML router handles the easy cases offline; LLM classifier handles ambiguity.
  - Every decision is logged so we can retrain the router on real traffic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config import (
    CONFIDENCE_THRESHOLD,
    LARGE_INPUT_THRESHOLD,
    ML_ROUTER_PATH,
    ML_ROUTER_CONFIDENCE_THRESHOLD,
    ENABLE_ML_ROUTER,
    ROUTING_LOG_PATH,
    CONVERSATION_WINDOW_SIZE,
)
from llm.small_llm import SmallLLM
from llm.large_llm import LargeLLM
from modules.classifier import Classifier
from modules.answerer import SimpleAnswerer
from modules.reasoner import Reasoner
from modules.personalizer import Personalizer
from modules.profile_updater import ProfileUpdater
from db.profile_db import ProfileDB
from memory import ConversationBuffer
from router import MLRouter, ClassificationLogger

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    """Structured response returned by the orchestrator for every query."""
    answer: str
    routing_decision: str        # "small" | "big" | "pre-flight:large-input"
    routed_by: str               # "ml" | "llm" | "pre-flight" | "none"
    model_used: str
    context_note: str
    classification: dict = field(default_factory=dict)
    ml_prediction: dict = field(default_factory=dict)    # {decision, confidence, probs}
    latency_ms: float = 0.0
    buffer_state: dict = field(default_factory=dict)
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

        # New: ML router (optional) + per-user memory + routing log
        self.ml_router: MLRouter | None = self._load_ml_router()
        self.buffers: dict[str, ConversationBuffer] = {}
        self.routing_log = ClassificationLogger(ROUTING_LOG_PATH)

        logger.info("Small LLM : %s", self.small_llm.model)
        logger.info("Large LLM : %s", self.large_llm.model)
        logger.info(
            "ML router : %s",
            "loaded" if self.ml_router else "disabled/not-found (using LLM classifier only)",
        )
        logger.info("Window size: %d turn-pairs", CONVERSATION_WINDOW_SIZE)
        logger.info("System ready.")

    # ── Profile helpers ───────────────────────────────────────────────────────

    def load_profile(self, user_id: str) -> dict:
        return self.profile_db.get(user_id)

    def end_session(self, user_id: str, session_queries: list[str]) -> dict:
        if not session_queries:
            return self.profile_db.get(user_id)

        logger.info("Updating profile for user '%s' (%d queries)...",
                    user_id, len(session_queries))

        current_profile = self.profile_db.get(user_id)
        updates = self.profile_updater.extract_updates(session_queries, current_profile)
        logger.debug("Extracted updates: %s", updates)

        updated_profile = self.profile_db.merge_updates(user_id, updates)
        logger.info("Profile updated. Interaction #%d",
                    updated_profile["interaction_count"])
        return updated_profile

    # ── Buffer helpers ────────────────────────────────────────────────────────

    def get_buffer(self, user_id: str) -> ConversationBuffer:
        """Return (creating if needed) this user's conversation buffer."""
        if user_id not in self.buffers:
            self.buffers[user_id] = ConversationBuffer(
                small_llm=self.small_llm,
                window_size=CONVERSATION_WINDOW_SIZE,
            )
        return self.buffers[user_id]

    def clear_buffer(self, user_id: str) -> None:
        if user_id in self.buffers:
            self.buffers[user_id].clear()

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
        Process a user query end-to-end.

        If `user_id` is provided, the profile is auto-loaded AND the per-user
        ConversationBuffer is used for history (the `history` arg is ignored).
        If no `user_id` is provided, the orchestrator falls back to the passed
        `history` list (legacy behavior — kept for backwards compatibility).
        """
        if user_id and not user_profile:
            user_profile = self.profile_db.get(user_id)

        # Per-user buffer takes precedence over passed-in history
        buffer = self.get_buffer(user_id) if user_id else None
        effective_history = buffer.build_history() if buffer else history

        start = time.time()

        try:
            # ── Step 1: Pre-flight ─────────────────────────────────────────
            if len(query.split()) > LARGE_INPUT_THRESHOLD:
                logger.info("Pre-flight: long query — routing directly to Large LLM")
                resp = self._route_big(
                    query, effective_history, user_profile,
                    "pre-flight:large-input", "pre-flight", start,
                )
                self._finalize(user_id, buffer, query, resp)
                return resp

            # ── Step 2: Route (ML → LLM cascade) ───────────────────────────
            routing, routed_by, classification, ml_pred = self._decide_route(query)

            # ── Step 3: Dispatch ───────────────────────────────────────────
            if routing == "small":
                answer = self.answerer.answer(query, effective_history, user_profile)
                resp = OrchestratorResponse(
                    answer=answer,
                    routing_decision="small",
                    routed_by=routed_by,
                    model_used=self.small_llm.model,
                    context_note="",
                    classification=classification,
                    ml_prediction=ml_pred,
                    latency_ms=self._ms(start),
                )
            else:
                resp = self._route_big(
                    query, effective_history, user_profile,
                    "big", routed_by, start, classification, ml_pred,
                )

            self._finalize(user_id, buffer, query, resp)
            return resp

        except Exception as exc:
            logger.exception("Error processing query: %s", exc)
            return OrchestratorResponse(
                answer=f"Error: {exc}",
                routing_decision="error",
                routed_by="none",
                model_used="none",
                context_note="",
                latency_ms=self._ms(start),
                error=str(exc),
            )

    # ── Routing (cascade) ─────────────────────────────────────────────────────

    def _decide_route(self, query: str) -> tuple[str, str, dict, dict]:
        """
        Decide routing for `query`. Returns:
            (routing_decision, routed_by, llm_classification_dict, ml_prediction_dict)

        Strategy:
          1. If ML router is loaded and enabled, use it.
          2. If ML confidence ≥ threshold, skip the LLM classifier.
          3. Otherwise, fall back to the LLM classifier.
        """
        ml_pred: dict = {}

        if self.ml_router is not None:
            try:
                pred = self.ml_router.predict(query)
                ml_pred = {
                    "decision": pred.decision,
                    "confidence": round(pred.confidence, 3),
                    "probs": {k: round(v, 3) for k, v in pred.probs.items()},
                }
                logger.info(
                    "ML router: %s (conf=%.2f, probs=%s)",
                    pred.decision, pred.confidence, ml_pred["probs"],
                )
                if pred.confidence >= ML_ROUTER_CONFIDENCE_THRESHOLD:
                    return pred.decision, "ml", {}, ml_pred
                logger.info(
                    "ML confidence %.2f < %.2f — cascading to LLM classifier",
                    pred.confidence, ML_ROUTER_CONFIDENCE_THRESHOLD,
                )
            except Exception as exc:                          # pragma: no cover
                logger.warning("ML router failed (%s) — falling back to LLM classifier.", exc)

        # Fallback / cascade: LLM classifier
        clf = self.classifier.classify(query)
        logger.info(
            "LLM classifier: %s (complexity=%s, conf=%.2f, intent=%s)",
            clf.routing_decision, clf.complexity, clf.confidence, clf.intent,
        )

        routing = clf.routing_decision
        if routing == "small" and clf.confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                "LLM confidence %.2f < %.2f — upgrading to Large LLM",
                clf.confidence, CONFIDENCE_THRESHOLD,
            )
            routing = "big"

        return routing, "llm", clf.raw, ml_pred

    # ── Big-route dispatch ────────────────────────────────────────────────────

    def _route_big(
        self,
        query: str,
        history: list[dict] | None,
        user_profile: dict | None,
        routing: str,
        routed_by: str,
        start: float,
        classification: dict | None = None,
        ml_prediction: dict | None = None,
    ) -> OrchestratorResponse:
        """
        Big-LLM flow:
          1. Large LLM produces the core answer (no profile)
          2. Small LLM summarizes it in 1 line (prevents context bloat)
          3. Small LLM produces personalized intro + closing using
             (query + profile + 1-line summary)
          4. Final = intro + core + closing
        """
        core_answer = self.reasoner.reason(query, history)

        answer_summary = self.context_adder.summarize(core_answer)
        logger.debug("Answer summary: %s", answer_summary)

        context = self.context_adder.add_context(query, user_profile, answer_summary)
        intro = context.get("intro", "")
        closing = context.get("closing", "")

        parts: list[str] = []
        if intro:
            parts.extend([intro, ""])
        parts.append(core_answer)
        if closing:
            parts.extend(["", "---", f"💡 {closing}"])

        final_answer = "\n".join(parts)

        return OrchestratorResponse(
            answer=final_answer,
            routing_decision=routing,
            routed_by=routed_by,
            model_used=self.large_llm.model,
            context_note=closing,
            classification=classification or {},
            ml_prediction=ml_prediction or {},
            latency_ms=self._ms(start),
        )

    # ── Post-turn bookkeeping ────────────────────────────────────────────────

    def _finalize(
        self,
        user_id: str | None,
        buffer: ConversationBuffer | None,
        query: str,
        resp: OrchestratorResponse,
    ) -> None:
        """Record the turn in memory + log the routing decision for retraining."""
        if buffer is not None:
            buffer.add_turn(query, resp.answer)
            resp.buffer_state = buffer.snapshot()

        self.routing_log.log(
            query=query,
            final_routing=resp.routing_decision,
            user_id=user_id,
            ml_decision=resp.ml_prediction.get("decision") if resp.ml_prediction else None,
            ml_confidence=resp.ml_prediction.get("confidence") if resp.ml_prediction else None,
            llm_decision=resp.classification.get("routing_decision") if resp.classification else None,
            llm_confidence=resp.classification.get("confidence") if resp.classification else None,
            model_used=resp.model_used,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_ml_router(self) -> MLRouter | None:
        if not ENABLE_ML_ROUTER:
            return None
        try:
            return MLRouter.load(ML_ROUTER_PATH)
        except FileNotFoundError:
            logger.warning(
                "ML router model not found at %s — run `python -m router.train`. "
                "Falling back to LLM classifier only.", ML_ROUTER_PATH,
            )
            return None
        except Exception as exc:                              # pragma: no cover
            logger.warning("Failed to load ML router (%s) — falling back.", exc)
            return None

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.time() - start) * 1000, 1)
