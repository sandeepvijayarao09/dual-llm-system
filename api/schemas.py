"""Request/response models. /v1/chat/completions follows the OpenAI shape."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # "auto" runs the routing cascade; "small" / "large" force a model.
    model: Literal["auto", "small", "large"] = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    # When set, the server keeps this user's profile and conversation memory,
    # and only the last user message is read from `messages`.
    user: str | None = Field(default=None, max_length=255)
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMCallOut(BaseModel):
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    latency_ms: float
    cost_usd: float | None
    error: str | None


class Routing(BaseModel):
    """Non-standard extension: how the cascade handled this request."""
    decision: str
    routed_by: str
    model_used: str
    latency_ms: float
    context_note: str
    ml_prediction: dict
    classification: dict
    cost_usd: float | None
    prices_as_of: str | None
    cached_input_tokens: int
    calls: list[LLMCallOut]
    buffer: dict
    prompts_sent: dict
    error: str | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    routing: Routing


class Persona(BaseModel):
    user_id: str
    label: str
    emoji: str


class SessionRequest(BaseModel):
    queries: list[str]


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    small_provider: str
    small_model: str
    large_provider: str
    large_model: str
    ml_router: bool
