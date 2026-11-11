"""
Small LLM client — the cheap, fast model (default: GPT-4o Mini).

Used for:
  - Query classification / routing
  - Direct answers to simple queries (personalized via user profile)
  - Adding short context notes after Large LLM answers
  - Profile updates at end of session
"""

from config import SMALL_LLM_TIMEOUT, SMALL_MODEL, SMALL_PROVIDER
from llm.base import ChatLLM


class SmallLLM(ChatLLM):
    role = "small"

    def __init__(self) -> None:
        super().__init__(SMALL_PROVIDER, SMALL_MODEL, SMALL_LLM_TIMEOUT)
