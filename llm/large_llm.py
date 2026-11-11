"""
Large LLM client — the capable, expensive model (default: GPT-4o).

Used for complex queries requiring deep reasoning.
Personalization is handled via system prompt injection (user profile).
"""

from config import LARGE_LLM_TIMEOUT, LARGE_MODEL, LARGE_PROVIDER
from llm.base import ChatLLM


class LargeLLM(ChatLLM):
    role = "large"

    def __init__(self) -> None:
        super().__init__(LARGE_PROVIDER, LARGE_MODEL, LARGE_LLM_TIMEOUT)

    def complete(
        self,
        system: str,
        user_message: str,
        history: list[dict] | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
        use_thinking: bool = True,
    ) -> str:
        return super().complete(system, user_message, history, max_tokens, json_mode)
