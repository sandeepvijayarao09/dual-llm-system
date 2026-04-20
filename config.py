"""
Configuration for the Dual LLM System.
- Large LLM  : GPT-4o / o-series via OpenAI API
- Small LLM  : GPT-4o Mini via OpenAI API
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI (both LLMs) ───────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LARGE_MODEL    = os.getenv("LARGE_MODEL",  "gpt-4o")
SMALL_MODEL    = os.getenv("SMALL_MODEL",  "gpt-4o-mini")

# ── Routing thresholds ───────────────────────────────────────────────────────
# Minimum classifier confidence to trust a "simple" routing decision
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# Token count above which we skip classification and go straight to Large LLM
LARGE_INPUT_THRESHOLD = int(os.getenv("LARGE_INPUT_THRESHOLD", "2000"))

# ── Timeouts (seconds) ───────────────────────────────────────────────────────
SMALL_LLM_TIMEOUT = int(os.getenv("SMALL_LLM_TIMEOUT", "60"))
LARGE_LLM_TIMEOUT = int(os.getenv("LARGE_LLM_TIMEOUT", "120"))

# ── Profile Database ──────────────────────────────────────────────────────────
# SQLite file path for storing user profiles
DB_PATH = os.getenv("DB_PATH", "profiles.db")

# ── ML Router ─────────────────────────────────────────────────────────────────
# Path to the fitted sklearn pipeline produced by `python -m router.train`
ML_ROUTER_PATH = os.getenv("ML_ROUTER_PATH", "router/models/router_v0.joblib")

# If ML router's top-class probability is below this, fall back to the
# LLM-based Classifier (cascade pattern). 0.65 is a sensible default given
# a modestly-sized seed dataset.
ML_ROUTER_CONFIDENCE_THRESHOLD = float(
    os.getenv("ML_ROUTER_CONFIDENCE_THRESHOLD", "0.65")
)

# When True and a model is on disk, the orchestrator uses the ML router first
# and only calls the LLM classifier on low-confidence cases. When False, the
# orchestrator uses the LLM classifier directly (old behavior).
ENABLE_ML_ROUTER = os.getenv("ENABLE_ML_ROUTER", "true").lower() == "true"

# SQLite log of every routing decision (used to bootstrap retraining data)
ROUTING_LOG_PATH = os.getenv("ROUTING_LOG_PATH", "routing_log.db")

# ── Conversation Buffer (sliding window) ─────────────────────────────────────
# Number of user+assistant PAIRS kept verbatim before being folded into a
# rolling summary. Increase for chats where follow-ups reach further back.
CONVERSATION_WINDOW_SIZE = int(os.getenv("CONVERSATION_WINDOW_SIZE", "3"))
