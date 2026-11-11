# API image: FastAPI + the routing cascade. Build: docker build -t dual-llm-api .
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --all-extras --no-install-project

FROM python:3.12-slim
RUN useradd --create-home --uid 1001 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY api ./api
COPY db ./db
COPY llm ./llm
COPY memory ./memory
COPY modules ./modules
COPY router ./router
COPY data/personas.json ./data/personas.json
COPY config.py orchestrator.py ./
RUN mkdir -p /app/var && chown app /app/var
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/app/var/profiles.db \
    ROUTING_LOG_PATH=/app/var/routing_log.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
