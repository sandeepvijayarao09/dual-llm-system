"""
Small LLM client — local model via Ollama REST API.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1
and its own native /api/chat endpoint. We use the native endpoint because
it doesn't require an extra SDK dependency.

Install Ollama:  https://ollama.com/download
Pull a model:    ollama pull llama3.2:3b
"""

import json
import requests
from config import OLLAMA_BASE_URL, SMALL_MODEL, SMALL_LLM_TIMEOUT


class SmallLLM:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model = SMALL_MODEL
        self._check_connection()

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user_message: str,
        history: list[dict] | None = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """
        Call the local Ollama model and return the response text.
        Set json_mode=True to request a JSON-formatted response.
        """
        messages = self._build_messages(system, history, user_message)

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1 if json_mode else 0.7,
            },
        }

        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=SMALL_LLM_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama timed out after {SMALL_LLM_TIMEOUT}s. "
                "Try a smaller model or increase SMALL_LLM_TIMEOUT."
            )

    def list_models(self) -> list[str]:
        """Return names of locally available Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_connection(self) -> None:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5)
        except Exception:
            print(
                f"⚠️  Warning: Cannot reach Ollama at {self.base_url}. "
                "Small LLM calls will fail until Ollama is started.\n"
                "  → Install : https://ollama.com/download\n"
                f"  → Pull    : ollama pull {self.model}\n"
                "  → Start   : ollama serve\n"
            )

    def _build_messages(
        self, system: str, history: list[dict] | None, user_message: str
    ) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages
