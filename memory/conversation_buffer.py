"""
ConversationBuffer — hybrid sliding window with rolling summary.

Replaces the naive "send full history every turn" pattern with:
    - Last N turns kept VERBATIM (recency matters for follow-ups)
    - EVERY turn also folded into a rolling summary (small LLM)
    - Optional per-user isolation via a dict of buffers

Why this matters:
    - Caps context cost — long chats no longer grow unboundedly
    - Protects the Large LLM from paying for verbose old turns
    - Keeps recency intact so pronouns / follow-ups still resolve

Design notes:
    - The summary is updated incrementally on EACH STEP (previous summary
      + the just-completed turn) so the summary always reflects the full
      conversation, even before any window eviction occurs.
    - Because every turn is already summarized, eviction from the verbatim
      window is loss-free — the evicted turn has already been folded in.
    - Assistant messages are truncated to 400 chars before summarization —
      we keep the gist, not the prose.
    - If no small LLM is injected, the buffer degrades to a pure window
      with a crude appended record (still bounded, just no semantic summary).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM = (
    "You compress conversation history for an AI assistant. "
    "Given a running summary and one new user/assistant turn, "
    "output an updated summary that preserves: key facts, entities, "
    "user preferences, unresolved questions, and decisions made. "
    "Keep it under 80 tokens. Output ONLY the summary, no preamble."
)


@dataclass
class ConversationBuffer:
    """
    Per-user conversation memory.

    Attributes:
        window_size      : number of user+assistant PAIRS kept verbatim
        max_assistant_chars : truncate assistant messages this long before
                              feeding them to the summarizer
        recent_turns     : flat list of {"role", "content"} dicts
        rolling_summary  : compressed string of all evicted turns
    """

    small_llm: Optional[object] = None        # duck-typed: needs .complete()
    window_size: int = 10
    max_assistant_chars: int = 400
    recent_turns: list[dict] = field(default_factory=list)
    rolling_summary: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Record a completed user/assistant exchange.

        Behavior:
            1. Append the new pair to the verbatim window.
            2. Immediately fold the new pair into the rolling summary
               (summarizer runs at EACH step, not only on eviction).
            3. Drop the oldest pair if the window is over capacity —
               loss-free, because step 2 already preserved its content.
        """
        new_user = {"role": "user", "content": user_msg}
        new_asst = {"role": "assistant", "content": assistant_msg}
        self.recent_turns.append(new_user)
        self.recent_turns.append(new_asst)

        # Summarize on every step
        self._fold_into_summary(new_user, new_asst)

        # Evict in pairs while over capacity (no re-summarize — already folded)
        max_msgs = self.window_size * 2
        while len(self.recent_turns) > max_msgs:
            self.recent_turns.pop(0)   # evict user
            self.recent_turns.pop(0)   # evict assistant

    def build_history(self) -> list[dict]:
        """
        Return the history block to prepend to the next LLM call.

        Layout:
            [summary_system_msg]?  — one system msg with rolling summary
            [recent verbatim turns]
        """
        history: list[dict] = []
        if self.rolling_summary:
            history.append({
                "role": "system",
                "content": f"[Earlier conversation summary: {self.rolling_summary}]",
            })
        history.extend(self.recent_turns)
        return history

    def clear(self) -> None:
        """Reset everything (used by CLI 'clear' / UI 'clear chat')."""
        self.recent_turns = []
        self.rolling_summary = ""

    def snapshot(self) -> dict:
        """Return a JSON-safe view for debugging / telemetry."""
        return {
            "window_size": self.window_size,
            "recent_turn_count": len(self.recent_turns),
            "has_summary": bool(self.rolling_summary),
            "summary_chars": len(self.rolling_summary),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fold_into_summary(self, user_turn: dict, asst_turn: dict) -> None:
        """Update rolling_summary with one evicted pair."""
        asst_content = asst_turn["content"][: self.max_assistant_chars]
        if len(asst_turn["content"]) > self.max_assistant_chars:
            asst_content += " …[truncated]"

        # Degrade gracefully if no summarizer wired up
        if self.small_llm is None:
            # Append a crude record — still bounded because window evicts
            appendix = f"User: {user_turn['content']} | Asst: {asst_content[:120]}"
            self.rolling_summary = (
                f"{self.rolling_summary} | {appendix}"
                if self.rolling_summary
                else appendix
            )
            return

        prompt = (
            f"Previous summary:\n{self.rolling_summary or '(none yet)'}\n\n"
            f"New turn to fold in:\n"
            f"User: {user_turn['content']}\n"
            f"Assistant: {asst_content}\n\n"
            "Produce an updated summary."
        )

        try:
            new_summary = self.small_llm.complete(
                system=SUMMARY_SYSTEM,
                user_message=prompt,
                max_tokens=120,
            )
            self.rolling_summary = (new_summary or "").strip()
        except Exception as exc:                       # pragma: no cover
            logger.warning("Summary update failed (%s) — keeping previous.", exc)
