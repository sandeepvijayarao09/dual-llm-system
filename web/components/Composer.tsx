"use client";

import { useRef, useState } from "react";
import type { ChatModel } from "@/lib/api";

interface Props {
  disabled: boolean;
  pending: boolean;
  model: ChatModel;
  onModelChange: (m: ChatModel) => void;
  onSend: (text: string) => void;
}

const MODELS: { value: ChatModel; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "small", label: "Small" },
  { value: "large", label: "Large" },
];

export default function Composer({ disabled, pending, model, onModelChange, onSend }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const canSend = !disabled && !pending && text.trim().length > 0;

  const submit = () => {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
    ref.current?.focus();
  };

  return (
    <div className="border-t border-border bg-panel px-4 py-3 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <form
          className="flex items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            ref={ref}
            aria-label="Message"
            rows={1}
            value={text}
            disabled={disabled}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask anything..."
            className="field-sizing-content max-h-48 min-h-10 min-w-0 flex-1 resize-none rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <select
            aria-label="Model"
            value={model}
            onChange={(e) => onModelChange(e.target.value as ChatModel)}
            className="h-10 rounded-md border border-border bg-bg px-2 text-sm"
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <button
            type="submit"
            disabled={!canSend}
            className="h-10 rounded-md bg-accent px-4 text-sm font-medium text-accent-fg disabled:opacity-40"
          >
            {pending ? "Sending" : "Send"}
          </button>
        </form>
        <p className="mt-1.5 text-xs text-muted">
          Enter to send, Shift+Enter for a new line. Prefix with <code className="font-mono">/small</code> or{" "}
          <code className="font-mono">/large</code> to force a model.
        </p>
      </div>
    </div>
  );
}
