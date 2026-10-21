/**
 * Typed client for the dual-llm-system FastAPI backend.
 *
 * Types mirror the HTTP API contract exactly. All requests go through the
 * Next.js rewrite `/api/:path*` -> `${API_URL}/:path*`.
 */

const BASE = "/api";

// ── Types ───────────────────────────────────────────────────────────────────

export interface Health {
  status: "ok";
  small_model: string;
  large_model: string;
  small_provider: string;
  large_provider: string;
  ml_router: boolean;
}

export interface Persona {
  user_id: string;
  label: string;
  emoji: string;
}

export type Expertise = "novice" | "intermediate" | "expert";

export interface Profile {
  user_id: string;
  name: string;
  expertise: Expertise;
  tone: string;
  domain: string;
  interests: string[];
  topics_discussed: string[];
  preferred_format: string;
  background?: string;
  interaction_count: number;
  last_updated: string;
}

export interface UsageBucket {
  queries: number;
  cost_usd: number | null;
  unpriced: number;
  input_tokens: number;
  output_tokens: number;
  avg_latency_ms: number | null;
}

export interface UsageSummary extends UsageBucket {
  by_route: {
    small: UsageBucket;
    big: UsageBucket;
  };
}

export type ChatModel = "auto" | "small" | "large";
export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatCompletionRequest {
  model: ChatModel;
  messages: ChatMessage[];
  user?: string;
}

export type RoutingDecision =
  | "small"
  | "big"
  | "forced:small"
  | "forced:large"
  | "pre-flight:large-input"
  | "error";

export type RoutedBy = "ml" | "llm" | "pre-flight" | "slash" | "none";

export interface MlPrediction {
  decision: string;
  confidence: number;
  probs: { small: number; big: number };
  latency_ms: number;
}

export interface LlmCall {
  stage: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  latency_ms: number;
  cost_usd: number | null;
  error: string | null;
}

export interface BufferState {
  window_size: number;
  recent_turn_count: number;
  has_summary: boolean;
  summary_chars: number;
}

export interface PromptSent {
  model: string;
  system: string;
  user: string;
}

/** `{}` when the stage did not run; TS models that as a partial-free empty object. */
type Empty = Record<string, never>;

export interface Routing {
  decision: RoutingDecision;
  routed_by: RoutedBy;
  model_used: string;
  latency_ms: number;
  context_note: string;
  ml_prediction: Empty | MlPrediction;
  classification: Record<string, unknown>;
  cost_usd: number | null;
  prices_as_of: string;
  cached_input_tokens: number;
  calls: LlmCall[];
  buffer: Empty | BufferState;
  prompts_sent: Record<string, PromptSent>;
  error: string | null;
}

export interface ChatCompletionChoice {
  index: number;
  message: { role: "assistant"; content: string };
  finish_reason: string;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: ChatCompletionUsage;
  routing: Routing;
}

export function hasMlPrediction(p: Routing["ml_prediction"]): p is MlPrediction {
  return "confidence" in p;
}

export function hasBuffer(b: Routing["buffer"]): b is BufferState {
  return "window_size" in b;
}

// ── Errors ──────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      return typeof detail === "string" ? detail : JSON.stringify(detail);
    }
    return JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (e) {
    throw new ApiError(0, `Network error: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

// ── Endpoints ───────────────────────────────────────────────────────────────

export const api = {
  health: () => request<Health>("/healthz"),
  personas: () => request<Persona[]>("/v1/personas"),
  profile: (userId: string) => request<Profile>(`/v1/profiles/${enc(userId)}`),
  updateProfileFromSession: (userId: string, queries: string[]) =>
    request<Profile>(`/v1/profiles/${enc(userId)}/session`, {
      method: "POST",
      body: JSON.stringify({ queries }),
    }),
  resetProfile: (userId: string) =>
    request<void>(`/v1/profiles/${enc(userId)}`, { method: "DELETE" }),
  clearConversation: (userId: string) =>
    request<void>(`/v1/conversations/${enc(userId)}`, { method: "DELETE" }),
  usage: () => request<UsageSummary>("/v1/usage"),
  chat: (body: ChatCompletionRequest, signal?: AbortSignal) =>
    request<ChatCompletionResponse>("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
};
