"""
ProfileUpdater — runs on the Small LLM (GPT-4o Mini via OpenAI).

At the end of each session, analyzes only the user's queries
(never the LLM answers) and extracts profile signals:
  - new topics / interests
  - expertise level inferred from vocabulary
  - domain inferred from subjects discussed
  - preferred tone inferred from how they write

Input is always small (just the user's queries, ~100-300 tokens).
Output is a small JSON update dict (~50-100 tokens).
No large LLM output is ever processed here.
"""

import json
import re
from llm.small_llm import SmallLLM

UPDATER_SYSTEM = """You are a user profile analyst.

Given a list of questions a user asked in one session and their current profile,
extract signals to update their profile.

Output ONLY a JSON object with these fields (omit fields you cannot infer):
{
  "interests":        ["topic1", "topic2"],
  "topics_discussed": ["topic1", "topic2"],
  "expertise":        "novice" | "intermediate" | "expert",
  "tone":             "casual" | "formal" | "technical",
  "domain":           "primary domain as a short string",
  "preferred_format": "bullets" | "prose" | "plain"
}

Rules:
- interests: topics the user seems genuinely curious about (2-5 items max)
- topics_discussed: all subjects mentioned in queries (be specific)
- expertise: infer from vocabulary complexity and depth of questions
- tone: infer from how the user phrases their questions
- domain: the main professional or academic area (e.g. "AI/ML", "finance", "medicine")
- preferred_format: only include if clearly signalled in the queries
- If you cannot confidently infer a field, omit it entirely.

Output ONLY the raw JSON. No explanation."""


class ProfileUpdater:
    def __init__(self, small_llm: SmallLLM):
        self.llm = small_llm

    def extract_updates(
        self,
        session_queries: list[str],
        current_profile: dict | None = None,
    ) -> dict:
        """
        Analyze session queries and return a dict of profile updates.
        Never receives LLM answers — only the user's own queries.

        Args:
            session_queries : list of raw user query strings from this session
            current_profile : existing profile for context (optional)

        Returns:
            dict with partial profile updates to be merged by ProfileDB
        """
        if not session_queries:
            return {}

        queries_text = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(session_queries)
        )

        profile_context = ""
        if current_profile:
            profile_context = (
                f"\nCurrent profile: "
                f"expertise={current_profile.get('expertise', 'unknown')}, "
                f"domain={current_profile.get('domain', 'unknown')}, "
                f"interests={current_profile.get('interests', [])}"
            )

        prompt = (
            f"User queries from this session:\n{queries_text}"
            f"{profile_context}\n\n"
            "Extract profile update signals from these queries."
        )

        raw = self.llm.complete(
            system=UPDATER_SYSTEM,
            user_message=prompt,
            max_tokens=200,
            json_mode=True,
        )

        return self._parse(raw)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse(self, raw_text: str) -> dict:
        """Parse JSON from model output with safe fallback."""
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
            data = json.loads(clean)
            # Sanitize list fields
            for field in ("interests", "topics_discussed"):
                if field in data and not isinstance(data[field], list):
                    data[field] = []
            return data
        except (json.JSONDecodeError, ValueError):
            return {}
