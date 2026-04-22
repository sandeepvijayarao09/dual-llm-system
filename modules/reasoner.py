"""
Reasoner — runs on the Large LLM (GPT-4o via OpenAI API).
Used for complex queries that require deep reasoning.

The Large LLM focuses on the core concept with a lightweight audience hint
(expertise + domain + format only — no full profile dump).
Full personalization (intro/closing) is still handled by the Small LLM.
"""

from llm.large_llm import LargeLLM

REASONER_SYSTEM = """You are an expert assistant capable of deep reasoning, analysis, and synthesis.
Think carefully and provide accurate, well-structured responses.
When solving complex problems, show your reasoning clearly.
Be thorough but avoid unnecessary repetition."""


class Reasoner:
    def __init__(self, large_llm: LargeLLM) -> None:
        self.llm = large_llm

    def reason(
        self,
        query: str,
        history: list[dict] | None = None,
        use_thinking: bool = True,
        user_profile: dict | None = None,
    ) -> str:
        """
        Generate a deep, reasoned response using the Large LLM.
        A lightweight audience hint (expertise, domain, format) is injected
        so the answer depth and scope match the user — without biasing reasoning
        with full profile details.
        """
        system = REASONER_SYSTEM

        if user_profile:
            hints = []
            if user_profile.get("expertise"):
                hints.append(f"expertise={user_profile['expertise']}")
            if user_profile.get("domain"):
                hints.append(f"domain={user_profile['domain']}")
            if user_profile.get("preferred_format"):
                hints.append(f"format={user_profile['preferred_format']}")
            if hints:
                system += (
                    f"\n\nAudience hint: {', '.join(hints)}. "
                    "Tailor the depth, scope, and vocabulary of your answer accordingly. "
                    "Only cover what is relevant to this audience's domain."
                )

        return self.llm.complete(
            system=system,
            user_message=query,
            history=history,
            use_thinking=use_thinking,
            max_tokens=4096,
        )
