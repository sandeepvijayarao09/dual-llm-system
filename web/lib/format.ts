import type { Routing } from "./api";

export function fmtUsd(cost: number | null | undefined): string {
  return cost == null ? "n/a (unpriced model)" : `$${cost.toFixed(6)}`;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "n/a";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

export function fmtPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

export type BadgeKind = "small" | "large" | "forced" | "preflight" | "error";

export function routeBadge(r: Routing): { kind: BadgeKind; label: string } {
  switch (r.decision) {
    case "small":
      return { kind: "small", label: "Small LLM" };
    case "big":
      return { kind: "large", label: "Large LLM" };
    case "forced:small":
      return { kind: "forced", label: "Forced · Small LLM" };
    case "forced:large":
      return { kind: "forced", label: "Forced · Large LLM" };
    case "pre-flight:large-input":
      return { kind: "preflight", label: "Pre-flight · Large LLM" };
    case "error":
      return { kind: "error", label: "Error" };
  }
}
