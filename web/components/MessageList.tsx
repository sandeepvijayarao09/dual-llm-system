"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatEntry } from "./types";
import AnswerMeta from "./AnswerMeta";

export default function MessageList({ messages, pending }: { messages: ChatEntry[]; pending: boolean }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-5">
        {messages.length === 0 && !pending && (
          <div className="py-16 text-center text-sm text-muted">
            <p>Ask a question. Each query is routed to the small or large model.</p>
            <p className="mt-1">Routing details, cost and the exact prompts appear under every answer.</p>
          </div>
        )}

        {messages.map((m) => {
          if (m.role === "user") {
            return (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-lg bg-sunken px-3.5 py-2 text-sm whitespace-pre-wrap break-words">
                  {m.content}
                </div>
              </div>
            );
          }
          if (m.role === "error") {
            return (
              <div
                key={m.id}
                role="alert"
                className="rounded-lg border border-danger/40 px-3.5 py-2 text-sm text-danger"
              >
                <span className="font-medium">Request failed.</span> {m.content}
              </div>
            );
          }
          return (
            <article key={m.id} className="min-w-0">
              <div className="prose-answer text-[15px]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              </div>
              <AnswerMeta response={m.response} />
            </article>
          );
        })}

        {pending && (
          <div className="flex items-center gap-2 text-sm text-muted" aria-live="polite">
            <span className="inline-block size-2 animate-pulse rounded-full bg-muted" />
            Routing and generating...
          </div>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
