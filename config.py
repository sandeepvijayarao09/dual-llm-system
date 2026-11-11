"""
Configuration for the Dual LLM System.

Settings are typed and validated by pydantic-settings, read from the
environment or a `.env` file. Each model role (small / large) names a
provider and a model, so the cascade can run on OpenAI, Anthropic or
Gemini models without code changes:

    SMALL_PROVIDER=openai     SMALL_MODEL=gpt-4o-mini
    LARGE_PROVIDER=anthropic  LARGE_MODEL=claude-opus-5-5

The module-level constants below are the public interface; the rest of the
codebase imports them (`from config import SMALL_MODEL`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "anthropic", "gemini"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Providers and models ─────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    small_provider: Provider = "openai"
    small_model: str = "gpt-4o-mini"
    large_provider: Provider = "openai"
    large_model: str = "gpt-4o"

    # ── Routing thresholds ───────────────────────────────────────────────────
    # Minimum LLM-classifier confidence to trust a "simple" routing decision
    confidence_threshold: float = Field(0.75, ge=0, le=1)
    # Word count above which classification is skipped (straight to Large LLM)
    large_input_threshold: int = Field(2000, gt=0)

    # ── Timeouts (seconds) ───────────────────────────────────────────────────
    small_llm_timeout: int = 60
    large_llm_timeout: int = 120

    # ── Storage ──────────────────────────────────────────────────────────────
    # SQLAlchemy URL for both profiles and the routing log, e.g.
    # postgresql+psycopg://user:pass@db/dual_llm. When unset, each store uses
    # its own SQLite file (DB_PATH, ROUTING_LOG_PATH) as before.
    database_url: str = ""
    db_path: str = "profiles.db"
    routing_log_path: str = "routing_log.db"

    # ── ML router ────────────────────────────────────────────────────────────
    # Fitted sklearn pipeline produced by `python -m router.train`
    ml_router_path: str = "router/models/router_v0.joblib"
    # Below this top-class probability the LLM classifier decides (cascade)
    ml_router_confidence_threshold: float = Field(0.65, ge=0, le=1)
    # False → the LLM classifier decides every query (old behavior)
    enable_ml_router: bool = True

    # ── Conversation buffer (sliding window) ─────────────────────────────────
    # user+assistant PAIRS kept verbatim before folding into a rolling summary
    conversation_window_size: int = Field(3, ge=1)

    # ── HTTP API ─────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000"]

    def api_key_for(self, provider: Provider) -> str:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
        }[provider]


settings = Settings()

OPENAI_API_KEY = settings.openai_api_key
SMALL_PROVIDER = settings.small_provider
SMALL_MODEL = settings.small_model
LARGE_PROVIDER = settings.large_provider
LARGE_MODEL = settings.large_model

CONFIDENCE_THRESHOLD = settings.confidence_threshold
LARGE_INPUT_THRESHOLD = settings.large_input_threshold

SMALL_LLM_TIMEOUT = settings.small_llm_timeout
LARGE_LLM_TIMEOUT = settings.large_llm_timeout

DATABASE_URL = settings.database_url
DB_PATH = settings.db_path
ROUTING_LOG_PATH = settings.routing_log_path

ML_ROUTER_PATH = settings.ml_router_path
ML_ROUTER_CONFIDENCE_THRESHOLD = settings.ml_router_confidence_threshold
ENABLE_ML_ROUTER = settings.enable_ml_router

CONVERSATION_WINDOW_SIZE = settings.conversation_window_size
