"""
FastAPI service exposing the routing cascade.

    uvicorn api.main:app --reload

POST /v1/chat/completions is OpenAI-compatible, so any OpenAI client can use
the router as a drop-in model ("auto" | "small" | "large"); the response adds
a `routing` object with the decision, per-call tokens, latency and estimated
cost. The other routes serve the web UI (personas, profiles, usage).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Health,
    Persona,
    Routing,
    SessionRequest,
    Usage,
)
from config import settings
from db.profile_db import ProfileDB
from orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def create_app(orchestrator: Orchestrator | None = None) -> FastAPI:
    """Build the app. Tests pass an orchestrator wired with fake LLMs."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.orchestrator = orchestrator or Orchestrator()
        yield

    app = FastAPI(title="Dual LLM System", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # One request at a time per user: the conversation buffer is per-user state.
    user_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    def orch(request: Request) -> Orchestrator:
        return request.app.state.orchestrator

    @app.get("/healthz", response_model=Health)
    def healthz(o: Orchestrator = Depends(orch)) -> Health:
        return Health(
            small_provider=getattr(o.small_llm, "provider_name", "unknown"),
            small_model=o.small_llm.model,
            large_provider=getattr(o.large_llm, "provider_name", "unknown"),
            large_model=o.large_llm.model,
            ml_router=o.ml_router is not None,
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(
        req: ChatCompletionRequest, o: Orchestrator = Depends(orch),
    ) -> ChatCompletionResponse:
        if req.stream:
            raise HTTPException(400, "stream=true is not supported yet")
        turns = [m for m in req.messages if m.role != "system"]
        if not turns or turns[-1].role != "user":
            raise HTTPException(422, "the last message must have role 'user'")

        query = turns[-1].content
        if req.model != "auto":
            query = f"/{req.model} {query}"

        if req.user:
            with user_locks[req.user]:
                resp = o.process(query, user_id=req.user)
        else:
            history = [m.model_dump() for m in turns[:-1]]
            resp = o.process(query, history=history)

        if resp.error:
            # Details (which can quote credentials) stay in the server log.
            kind = resp.error.split(":", 1)[0] if ":" in resp.error else "error"
            raise HTTPException(502, f"LLM provider call failed ({kind})")

        u = resp.usage or {}
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=resp.model_used,
            choices=[Choice(message=ChatMessage(role="assistant", content=resp.answer))],
            usage=Usage(
                prompt_tokens=u.get("input_tokens", 0),
                completion_tokens=u.get("output_tokens", 0),
                total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
            ),
            routing=Routing(
                decision=resp.routing_decision,
                routed_by=resp.routed_by,
                model_used=resp.model_used,
                latency_ms=resp.latency_ms,
                context_note=resp.context_note,
                ml_prediction=resp.ml_prediction,
                classification=resp.classification,
                cost_usd=u.get("cost_usd"),
                prices_as_of=u.get("prices_as_of"),
                cached_input_tokens=u.get("cached_input_tokens", 0),
                calls=u.get("calls", []),
                buffer=resp.buffer_state,
                prompts_sent=resp.prompts_sent,
                error=resp.error,
            ),
        )

    @app.get("/v1/personas", response_model=list[Persona])
    def personas() -> list[Persona]:
        return [Persona(**{k: p[k] for k in ("user_id", "label", "emoji")})
                for p in ProfileDB.personas()]

    @app.get("/v1/profiles/{user_id}")
    def get_profile(user_id: str, o: Orchestrator = Depends(orch)) -> dict:
        return o.load_profile(user_id)

    @app.post("/v1/profiles/{user_id}/session")
    def end_session(user_id: str, body: SessionRequest, o: Orchestrator = Depends(orch)) -> dict:
        try:
            return o.end_session(user_id, body.queries)
        except Exception as exc:          # provider failure inside the updater
            logger.exception("Profile update failed")
            raise HTTPException(502, f"Profile update failed ({type(exc).__name__})") from exc

    @app.delete("/v1/profiles/{user_id}", status_code=204)
    def reset_profile(user_id: str, o: Orchestrator = Depends(orch)) -> Response:
        """Back to the default profile — or, for a demo persona, its seed."""
        o.profile_db.delete(user_id)
        o.profile_db.seed_personas()
        return Response(status_code=204)

    @app.delete("/v1/conversations/{user_id}", status_code=204)
    def clear_conversation(user_id: str, o: Orchestrator = Depends(orch)) -> Response:
        o.clear_buffer(user_id)
        return Response(status_code=204)

    @app.get("/v1/usage")
    def usage(o: Orchestrator = Depends(orch)) -> dict:
        return o.routing_log.cost_summary()

    return app


app = create_app()
