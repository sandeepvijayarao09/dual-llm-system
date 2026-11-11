import type { ChatCompletionResponse } from "@/lib/api";

export type ChatEntry =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: string; response: ChatCompletionResponse }
  | { id: string; role: "error"; content: string };

export interface Notice {
  kind: "ok" | "warn" | "error";
  text: string;
}

export interface SessionCost {
  total: number;
  answers: number;
  unpriced: number;
}
