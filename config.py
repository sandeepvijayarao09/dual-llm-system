"""
Configuration for the Dual LLM System.

Settings are typed and validated by pydantic-settings, read from the
environment or a `.env` file. Each model role (small / large) names a
provider and a model. Defaults: Gemma 4 E4B running locally through Ollama
answers the simple queries and runs every support task; Gemini answers the
hard ones.

    SMALL_PROVIDER=ollama   SMALL_MODEL=gemma4:e4b
    LARGE_PROVIDER=gemini   LARGE_MODEL=gemini-3.8-flash
    # free alternative for the large role (Gemini API free tier, rate-limited):
    LARGE_PROVIDER=gemini   LARGE_MODEL=gemma-4-31b-it

The module-level constants below are the public interface; the rest of the
codebase imports them (`from config import SMALL_MODEL`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "gemini"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Providers and models ─────────────────────────────────────────────────
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    small_provider: Provider = "ollama"
    small_model: str = "gemma4:e4b"
    large_provider: Provider = "gemini"
    large_model: str = "gemini-3.8-flash"

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
    # Fitted sklearn pipeline. v1 (router/refine.py) is the default; v0 is the
    # published model the paper's numbers come from (router/train.py).
    ml_router_path: str = "router/models/router_v1.joblib"
    # Below this top-class probability the LLM classifier decides (cascade)
    ml_router_confidence_threshold: float = Field(0.65, ge=0, le=1)
    # False → the LLM classifier decides every query (old behavior)
    enable_ml_router: bool = True

    # ── Conversation buffer (sliding window) ─────────────────────────────────
    # user+assistant PAIRS kept verbatim before folding into a rolling summary
    conversation_window_size: int = Field(3, ge=1)

    # ── HTTP API ─────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()

GEMINI_API_KEY = settings.gemini_api_key
OLLAMA_BASE_URL = settings.ollama_base_url
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
