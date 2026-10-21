"use client";

import type { Persona, Profile } from "@/lib/api";
import { fmtUsd } from "@/lib/format";
import type { Notice, SessionCost } from "./types";

interface Props {
  personas: Persona[];
  personasError: string | null;
  userId: string | null;
  onSelect: (id: string) => void;
  profile: Profile | null;
  profileError: string | null;
  cost: SessionCost;
  sessionQueryCount: number;
  isGuest: boolean;
  busyAction: string | null;
  notice: Notice | null;
  onDismissNotice: () => void;
  onClear: () => void;
  onUpdate: () => void;
  onReset: () => void;
}

const noticeTone: Record<Notice["kind"], string> = {
  ok: "border-ok/40 text-ok",
  warn: "border-warn/40 text-warn",
  error: "border-danger/40 text-danger",
};

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h2 className="mb-2 text-[11px] font-semibold tracking-wider text-muted uppercase">{children}</h2>;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[6.5rem_1fr] gap-2 py-1 text-sm">
      <dt className="text-muted">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  );
}

function Chips({ items, empty }: { items: string[]; empty: string }) {
  if (items.length === 0) return <span className="text-muted">{empty}</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {items.map((t) => (
        <span key={t} className="rounded border border-border bg-sunken px-1.5 py-0.5 text-xs">
          {t}
        </span>
      ))}
    </span>
  );
}

function ActionButton({
  onClick,
  disabled,
  busy,
  children,
  title,
  danger,
}: {
  onClick: () => void;
  disabled?: boolean;
  busy?: boolean;
  children: React.ReactNode;
  title?: string;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || busy}
      title={title}
      className={`w-full rounded-md border border-border px-3 py-1.5 text-left text-sm transition-colors hover:bg-sunken disabled:cursor-not-allowed disabled:opacity-50 ${
        danger ? "text-danger" : ""
      }`}
    >
      {busy ? "Working..." : children}
    </button>
  );
}

export default function Sidebar(p: Props) {
  const { profile } = p;
  const anyBusy = p.busyAction !== null;

  return (
    <div className="flex flex-col gap-6 p-5">
      <section>
        <SectionTitle>Persona</SectionTitle>
        {p.personasError ? (
          <p className="text-sm text-danger">Could not load personas: {p.personasError}</p>
        ) : (
          <select
            aria-label="Select a persona"
            className="w-full rounded-md border border-border bg-panel px-2.5 py-1.5 text-sm"
            value={p.userId ?? ""}
            onChange={(e) => p.onSelect(e.target.value)}
            disabled={p.personas.length === 0}
          >
            {p.personas.length === 0 && <option value="">Loading...</option>}
            {p.personas.map((persona) => (
              <option key={persona.user_id} value={persona.user_id}>
                {persona.label}
              </option>
            ))}
          </select>
        )}
      </section>

      <section>
        <SectionTitle>Profile</SectionTitle>
        {p.profileError ? (
          <p className="text-sm text-danger">Could not load profile: {p.profileError}</p>
        ) : !profile ? (
          <p className="text-sm text-muted">Loading...</p>
        ) : (
          <>
            <dl>
              <Field label="Name">{profile.name || "Guest"}</Field>
              <Field label="Expertise">{profile.expertise}</Field>
              <Field label="Tone">{profile.tone || <span className="text-muted">not set</span>}</Field>
              <Field label="Domain">{profile.domain || <span className="text-muted">not set</span>}</Field>
              <Field label="Interests">
                <Chips items={profile.interests} empty="none" />
              </Field>
              <Field label="Sessions">{profile.interaction_count}</Field>
              <Field label="Topics">
                <Chips items={profile.topics_discussed} empty="none yet" />
              </Field>
            </dl>
            {profile.background && (
              <p className="mt-3 border-l-2 border-border pl-3 text-sm text-muted italic">{profile.background}</p>
            )}
          </>
        )}
      </section>

      <section>
        <SectionTitle>Session cost (est.)</SectionTitle>
        <p className="text-2xl font-semibold tabular-nums">{fmtUsd(p.cost.total)}</p>
        <p className="mt-1 text-xs text-muted">
          {p.cost.answers === 0
            ? "No answers yet in this chat."
            : `Across ${p.cost.answers} answer${p.cost.answers === 1 ? "" : "s"}, from token usage and list prices.`}
        </p>
        {p.cost.unpriced > 0 && (
          <p className="mt-1 text-xs text-warn">
            {p.cost.unpriced} answer{p.cost.unpriced === 1 ? "" : "s"} used an unpriced model and{" "}
            {p.cost.unpriced === 1 ? "is" : "are"} excluded.
          </p>
        )}
      </section>

      <section>
        <SectionTitle>Session</SectionTitle>
        <div className="flex flex-col gap-2">
          <ActionButton onClick={p.onClear} disabled={!p.userId || anyBusy} busy={p.busyAction === "clear"}>
            Clear chat
          </ActionButton>
          <ActionButton
            onClick={p.onUpdate}
            disabled={!p.userId || p.isGuest || p.sessionQueryCount === 0 || anyBusy}
            busy={p.busyAction === "update"}
            title={p.isGuest ? "Guest sessions don't update a profile" : "Run the profile updater on this session's queries"}
          >
            Update profile from session
            <span className="ml-1 text-muted">({p.sessionQueryCount})</span>
          </ActionButton>
          <ActionButton
            onClick={p.onReset}
            disabled={!p.userId || p.isGuest || anyBusy}
            busy={p.busyAction === "reset"}
            title={p.isGuest ? "The guest profile cannot be reset" : "Reset the stored profile to its default"}
            danger
          >
            Reset profile
          </ActionButton>
        </div>
        {p.notice && (
          <div
            role="status"
            className={`mt-3 flex items-start gap-2 rounded-md border px-3 py-2 text-sm ${noticeTone[p.notice.kind]}`}
          >
            <span className="min-w-0 flex-1 break-words">{p.notice.text}</span>
            <button
              type="button"
              onClick={p.onDismissNotice}
              aria-label="Dismiss"
              className="text-muted hover:text-fg"
            >
              ×
            </button>
          </div>
        )}
      </section>
    </div>
  );
}
