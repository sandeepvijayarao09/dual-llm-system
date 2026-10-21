"use client";

import { hasBuffer, hasMlPrediction, type ChatCompletionResponse } from "@/lib/api";
import { fmtMs, fmtPct, fmtUsd, routeBadge, type BadgeKind } from "@/lib/format";

const badgeTone: Record<BadgeKind, string> = {
  small: "border-ok/40 text-ok",
  large: "border-accent/50 text-accent",
  forced: "border-warn/40 text-warn",
  preflight: "border-warn/40 text-warn",
  error: "border-danger/40 text-danger",
};

function Disclosure({ summary, children }: { summary: string; children: React.ReactNode }) {
  return (
    <details className="group rounded-md border border-border">
      <summary className="cursor-pointer list-none px-3 py-1.5 text-xs font-medium text-muted select-none hover:text-fg">
        <span className="mr-1.5 inline-block transition-transform group-open:rotate-90">›</span>
        {summary}
      </summary>
      <div className="border-t border-border px-3 py-3">{children}</div>
    </details>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[8rem_1fr] gap-3 py-0.5 text-xs">
      <dt className="text-muted">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  );
}

function Mono({ children }: { children: string }) {
  return (
    <pre className="max-h-80 overflow-auto rounded-md border border-border bg-sunken px-3 py-2 font-mono text-xs whitespace-pre-wrap break-words">
      {children}
    </pre>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <h4 className="mt-4 mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">{children}</h4>;
}

export default function AnswerMeta({ response }: { response: ChatCompletionResponse }) {
  const r = response.routing;
  const badge = routeBadge(r);
  const totalIn = r.calls.reduce((s, c) => s + c.input_tokens, 0);
  const totalOut = r.calls.reduce((s, c) => s + c.output_tokens, 0);
  const tokensIn = totalIn || response.usage.prompt_tokens;
  const tokensOut = totalOut || response.usage.completion_tokens;
  const ml = r.ml_prediction;
  const prompts = Object.entries(r.prompts_sent);
  const hasClassification = Object.keys(r.classification).length > 0;

  return (
    <div className="mt-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        <span className={`rounded border px-1.5 py-0.5 font-medium ${badgeTone[badge.kind]}`}>
          {badge.label}
          <span className="ml-1 font-mono font-normal opacity-80">{r.model_used || response.model}</span>
        </span>
        <span className="tabular-nums">{fmtMs(r.latency_ms)}</span>
        <span className="tabular-nums">{fmtUsd(r.cost_usd)}</span>
        <span className="tabular-nums">
          {tokensIn} in
          {r.cached_input_tokens > 0 ? ` (${r.cached_input_tokens} cached)` : ""} / {tokensOut} out
        </span>
      </div>

      {r.error && <p className="text-xs text-danger">Routing error: {r.error}</p>}

      <Disclosure summary="Routing details">
        <dl>
          <Row label="Decision">
            <span className="font-mono">{r.decision}</span>
          </Row>
          <Row label="Routed by">
            <span className="font-mono">{r.routed_by}</span>
          </Row>
          <Row label="Model">
            <span className="font-mono">{r.model_used}</span>
          </Row>
          <Row label="Latency">{fmtMs(r.latency_ms)}</Row>
          <Row label="Est. cost">
            {fmtUsd(r.cost_usd)}
            <span className="text-muted"> (list prices as of {r.prices_as_of || "?"})</span>
          </Row>
          <Row label="Context note">{r.context_note || <span className="text-muted">(none)</span>}</Row>
          <Row label="ML prediction">
            {hasMlPrediction(ml) ? (
              <span>
                <span className="font-mono">{ml.decision}</span> at {fmtPct(ml.confidence)} confidence
                <span className="text-muted">
                  {" "}
                  (small {fmtPct(ml.probs.small)}, big {fmtPct(ml.probs.big)}; {fmtMs(ml.latency_ms)})
                </span>
              </span>
            ) : (
              <span className="text-muted">(not used)</span>
            )}
          </Row>
          <Row label="Buffer">
            {hasBuffer(r.buffer) ? (
              <span>
                window {r.buffer.window_size}, {r.buffer.recent_turn_count} recent turn
                {r.buffer.recent_turn_count === 1 ? "" : "s"},{" "}
                {r.buffer.has_summary ? `summary ${r.buffer.summary_chars} chars` : "no summary"}
              </span>
            ) : (
              <span className="text-muted">(no buffer)</span>
            )}
          </Row>
        </dl>

        <Label>LLM classification</Label>
        {hasClassification ? (
          <Mono>{JSON.stringify(r.classification, null, 2)}</Mono>
        ) : (
          <p className="text-xs text-muted">(not used)</p>
        )}

        <Label>Calls</Label>
        {r.calls.length === 0 ? (
          <p className="text-xs text-muted">(none recorded)</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[32rem] border-collapse text-xs">
              <thead>
                <tr className="border-b border-border text-left text-muted">
                  <th className="py-1 pr-3 font-medium">Stage</th>
                  <th className="py-1 pr-3 font-medium">Model</th>
                  <th className="py-1 pr-3 text-right font-medium">Tokens in→out</th>
                  <th className="py-1 pr-3 text-right font-medium">Latency</th>
                  <th className="py-1 text-right font-medium">Cost</th>
                </tr>
              </thead>
              <tbody>
                {r.calls.map((c, i) => (
                  <tr key={`${c.stage}-${i}`} className="border-b border-border/60 align-top last:border-0">
                    <td className="py-1 pr-3 font-mono">
                      {c.stage}
                      {c.error && <div className="font-sans text-danger">error: {c.error}</div>}
                    </td>
                    <td className="py-1 pr-3 font-mono">{c.model}</td>
                    <td className="py-1 pr-3 text-right tabular-nums">
                      {c.input_tokens}→{c.output_tokens}
                      {c.cached_input_tokens > 0 && (
                        <span className="text-muted"> ({c.cached_input_tokens} cached)</span>
                      )}
                    </td>
                    <td className="py-1 pr-3 text-right tabular-nums">{fmtMs(c.latency_ms)}</td>
                    <td className="py-1 text-right tabular-nums">
                      {c.cost_usd == null ? <span className="text-muted">n/a</span> : fmtUsd(c.cost_usd)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {prompts.length > 0 && (
          <div className="mt-4">
            <Disclosure summary={`Prompts sent (${prompts.length})`}>
              <div className="flex flex-col gap-5">
                {prompts.map(([label, p]) => (
                  <div key={label}>
                    <p className="text-xs font-semibold">
                      {label} <span className="ml-1 font-mono font-normal text-muted">{p.model}</span>
                    </p>
                    <Label>System prompt</Label>
                    <Mono>{p.system}</Mono>
                    <Label>User message</Label>
                    <Mono>{p.user}</Mono>
                  </div>
                ))}
              </div>
            </Disclosure>
          </div>
        )}
      </Disclosure>
    </div>
  );
}
