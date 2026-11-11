# Changelog

## 2.0.0 — unreleased

The cascade now runs on Gemma 4 E4B (local, via Ollama) and Gemini, behind a
FastAPI service and a Next.js UI, with per-call cost accounting and a
refined router. Breaking: OpenAI and Anthropic are no longer supported, the
Streamlit app is gone, and configuration moved to typed settings
(`.env.example`).

**Results** (Eval-1000; routing accuracy, not answer quality — see FINDINGS.md)

- Full cascade with a real classifier: 94.7% (router v0) and **95.8%**
  (router v1, the new default; fixes 12 of v0's errors, adds 1, p = 0.003).
  The one-line word-count rule gets 89.7%.
- Estimated cost ≈ 49.6% of sending everything to Gemini (§7a).
- Deferral curve with random and oracle baselines (§10).
- JEV (OpenRouter) as router and as fallback classifier, vs Gemma and the
  ML router (§11).

**Added**

- `api/`: FastAPI service; `POST /v1/chat/completions` in the OpenAI
  chat-completions format (`model: "auto" | "small" | "large"`) with a
  `routing` object: decision, per-call tokens, latency, estimated cost and
  prompts. Endpoints for personas, profiles, conversations and usage.
- `web/`: Next.js + TypeScript + Tailwind UI with routing details,
  per-call cost table, session cost, and LaTeX rendering (KaTeX).
- `llm/providers.py`: Ollama and Gemini adapters behind one interface, with
  exponential-backoff retries on transient errors.
- `llm/usage.py`: per-call tokens, latency and estimated cost, logged to
  `routing_log.db` (`routing_events` + `llm_calls`).
- Router v1 (`router/distill.py`, `router/refine.py`): distilled from the
  Gemma classifier, weight chosen on an Eval-500 dev set.
- Evals: `router/cost_eval.py`, `router/live_cascade_eval.py`,
  `router/deferral_curve.py`, `router/jev_eval.py`.
- SQLAlchemy storage: SQLite by default, Postgres via `DATABASE_URL`.
- `pyproject.toml` + `uv.lock`, ruff, mypy, Dockerfiles, docker-compose,
  CI for lint, types, tests, eval reproduction, web build and both images.
- `docs/industry-comparison.md`: RouteLLM, FrugalGPT, RouterBench and
  commercial routers.

**Fixed**

- The small model invented user backgrounds when given only a name.
- `python -m router.train` overwrote the published `router_v0.joblib`.
- "$5 and $10" rendered as math.
- A missing SQLite folder crashed startup.

**Unchanged**

- `router_v0.joblib` and the published 92.6% (CI still checks it).

## 1.0.0 — 2026-11-10

Cascade router between GPT-4o-mini and GPT-4o: 92.6% routing accuracy on
Eval-1000 (95% CI 90.9–94.2), the controlled ablation in FINDINGS.md, and
CI on Python 3.11 and 3.12.
