"""
ContextAdder — runs on the Small LLM (GPT-4o Mini via OpenAI).

Wraps the Large LLM's core answer with personalized context:
  - INTRO: a personalized framing before the answer
  - CLOSING: a personalized note after the answer

KEY DESIGN:
  - Large LLM output is NOT passed in full (avoids context overflow).
  - Only receives: query + user profile + 1-line summary of the answer.
  - The model reads the profile and decides how to frame the answer.
"""

import json
import logging

from llm.small_llm import SmallLLM

logger = logging.getLogger(__name__)

CONTEXT_SYSTEM = """You are a personalisation assistant.

You will receive:
1. A user's profile (JSON)
2. Their query
3. A one-line summary of the answer they are about to read

Your job is to write TWO short sections:

INTRO (2-3 sentences):
- Frame the answer for this specific user
- Connect it to their background, domain, or interests
- Set the right context so the answer lands well for them

CLOSING (1-2 sentences):
- Add a personal takeaway or connection
- Relate the topic to something in their profile

Output format (strictly follow this):
INTRO: <your intro text>
CLOSING: <your closing text>

Rules:
- Do NOT repeat or summarise the main answer
- Do NOT answer the question yourself
- Keep both sections short and personal
- Output ONLY in the format above, nothing else"""


class Personalizer:
    """
    Wraps Large LLM answers with personalized intro and closing.
    Never receives the full Large LLM output — only a 1-line summary.
    """

    def __init__(self, small_llm: SmallLLM) -> None:
        self.llm = small_llm

    def add_context(
        self,
        query: str,
        user_profile: dict | None = None,
        answer_summary: str = "",
    ) -> dict:
        """
        Generate personalized intro and closing based on query,
        user profile, and a 1-line summary of the Large LLM answer.

        Returns:
            {"intro": "...", "closing": "..."} or empty strings if no profile.
        """
        if not user_profile:
            return {"intro": "", "closing": ""}

        prompt = (
            f"User profile:\n{json.dumps(user_profile, indent=2)}\n\n"
            f"User asked: {query}\n\n"
            f"Answer summary: {answer_summary}"
        )

        raw = self.llm.complete(
            system=CONTEXT_SYSTEM,
            user_message=prompt,
            max_tokens=250,
        )

        return self._parse(raw)

    def summarize(self, text: str) -> str:
        """
        Generate a 1-line summary of the Large LLM answer.
        Used to give the context adder awareness of the answer
        without passing the full output.
        """
        result = self.llm.complete(
            system="Summarize the following text in exactly one sentence. Output only the summary.",
            user_message=text,
            max_tokens=60,
        )
        return result.strip() if result else ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> dict:
        """Parse INTRO: and CLOSING: from model output."""
        intro = ""
        closing = ""

        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("INTRO:"):
                intro = line[len("INTRO:"):].strip()
            elif line.upper().startswith("CLOSING:"):
                closing = line[len("CLOSING:"):].strip()

        return {"intro": intro, "closing": closing}
