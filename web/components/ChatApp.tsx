"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type ChatModel, type Persona, type Profile } from "@/lib/api";
import type { ChatEntry, Notice } from "./types";
import Sidebar from "./Sidebar";
import MessageList from "./MessageList";
import Composer from "./Composer";

const GUEST = "guest";

function errText(e: unknown): string {
  if (e instanceof ApiError) return e.status ? `${e.status}: ${e.message}` : e.message;
  return e instanceof Error ? e.message : String(e);
}

let nextId = 0;
const newId = () => `m${++nextId}`;

export default function ChatApp() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [personasError, setPersonasError] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatEntry[]>([]);
  const [sessionQueries, setSessionQueries] = useState<string[]>([]);
  const [model, setModel] = useState<ChatModel>("auto");
  const [pending, setPending] = useState(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Bumped whenever the chat is reset, so late responses from a previous chat are dropped.
  const generation = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const loadProfile = useCallback(async (id: string) => {
    setProfileError(null);
    try {
      setProfile(await api.profile(id));
    } catch (e) {
      setProfile(null);
      setProfileError(errText(e));
    }
  }, []);

  useEffect(() => {
    api
      .personas()
      .then((list) => {
        setPersonas(list);
        const first = list[0];
        if (first) setUserId((cur) => cur ?? first.user_id);
      })
      .catch((e: unknown) => setPersonasError(errText(e)));
  }, []);

  useEffect(() => {
    if (userId) void loadProfile(userId);
  }, [userId, loadProfile]);

  const resetLocalChat = useCallback(() => {
    generation.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setPending(false);
    setMessages([]);
    setSessionQueries([]);
  }, []);

  const switchPersona = useCallback(
    (id: string) => {
      if (id === userId) return;
      if (userId) {
        // Fire-and-forget: drop the previous persona's server-side buffer.
        api.clearConversation(userId).catch(() => undefined);
      }
      resetLocalChat();
      setNotice(null);
      setProfile(null);
      setUserId(id);
      setDrawerOpen(false);
    },
    [userId, resetLocalChat],
  );

  const clearChat = useCallback(async () => {
    if (!userId) return;
    resetLocalChat();
    setBusyAction("clear");
    try {
      await api.clearConversation(userId);
      setNotice({ kind: "ok", text: "Chat cleared." });
    } catch (e) {
      setNotice({ kind: "error", text: `Could not clear server buffer: ${errText(e)}` });
    } finally {
      setBusyAction(null);
    }
  }, [userId, resetLocalChat]);

  const updateProfile = useCallback(async () => {
    if (!userId) return;
    if (sessionQueries.length === 0) {
      setNotice({ kind: "warn", text: "No queries in this session to analyze yet." });
      return;
    }
    setBusyAction("update");
    try {
      const updated = await api.updateProfileFromSession(userId, sessionQueries);
      setProfile(updated);
      setSessionQueries([]);
      const interests = updated.interests.length ? updated.interests.join(", ") : "none";
      setNotice({
        kind: "ok",
        text: `Profile updated. Expertise: ${updated.expertise}. Interests: ${interests}.`,
      });
    } catch (e) {
      setNotice({ kind: "error", text: `Profile update failed: ${errText(e)}` });
    } finally {
      setBusyAction(null);
    }
  }, [userId, sessionQueries]);

  const resetProfile = useCallback(async () => {
    if (!userId || userId === GUEST) return;
    if (!window.confirm(`Reset the stored profile for "${userId}" to its default?`)) return;
    setBusyAction("reset");
    try {
      await api.resetProfile(userId);
      await loadProfile(userId);
      setNotice({ kind: "ok", text: `Profile for "${userId}" reset.` });
    } catch (e) {
      setNotice({ kind: "error", text: `Reset failed: ${errText(e)}` });
    } finally {
      setBusyAction(null);
    }
  }, [userId, loadProfile]);

  const send = useCallback(
    async (content: string) => {
      if (!userId || pending) return;
      const gen = generation.current;
      const controller = new AbortController();
      abortRef.current = controller;

      setMessages((m) => [...m, { id: newId(), role: "user", content }]);
      setSessionQueries((q) => [...q, content]);
      setPending(true);

      try {
        const res = await api.chat(
          { model, user: userId, messages: [{ role: "user", content }] },
          controller.signal,
        );
        if (gen !== generation.current) return;
        const answer = res.choices[0]?.message.content ?? "";
        setMessages((m) => [...m, { id: newId(), role: "assistant", content: answer, response: res }]);
      } catch (e) {
        if (gen !== generation.current) return;
        setMessages((m) => [...m, { id: newId(), role: "error", content: errText(e) }]);
      } finally {
        if (gen === generation.current) {
          setPending(false);
          abortRef.current = null;
        }
      }
    },
    [userId, pending, model],
  );

  const cost = useMemo(() => {
    let total = 0;
    let answers = 0;
    let unpriced = 0;
    for (const m of messages) {
      if (m.role !== "assistant") continue;
      answers += 1;
      if (m.response.routing.cost_usd == null) unpriced += 1;
      else total += m.response.routing.cost_usd;
    }
    return { total, answers, unpriced };
  }, [messages]);

  const sidebar = (
    <Sidebar
      personas={personas}
      personasError={personasError}
      userId={userId}
      onSelect={switchPersona}
      profile={profile}
      profileError={profileError}
      cost={cost}
      sessionQueryCount={sessionQueries.length}
      isGuest={userId === GUEST}
      busyAction={busyAction}
      notice={notice}
      onDismissNotice={() => setNotice(null)}
      onClear={clearChat}
      onUpdate={updateProfile}
      onReset={resetProfile}
    />
  );

  return (
    <div className="flex h-dvh overflow-hidden">
      {/* Desktop sidebar */}
      <aside className="hidden w-80 shrink-0 overflow-y-auto border-r border-border bg-panel lg:block">
        {sidebar}
      </aside>

      {/* Mobile drawer */}
      {drawerOpen && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
          <button
            type="button"
            aria-label="Close sidebar"
            className="absolute inset-0 bg-black/40"
            onClick={() => setDrawerOpen(false)}
          />
          <aside className="absolute inset-y-0 left-0 w-[min(20rem,85vw)] overflow-y-auto border-r border-border bg-panel shadow-xl">
            {sidebar}
          </aside>
        </div>
      )}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-border px-4 py-3 sm:px-6">
          <button
            type="button"
            className="rounded-md border border-border px-2.5 py-1 text-sm lg:hidden"
            onClick={() => setDrawerOpen(true)}
          >
            Menu
          </button>
          <div className="min-w-0">
            <h1 className="text-base font-semibold">Dual LLM System</h1>
            <p className="truncate text-xs text-muted">
              {profile
                ? `Chatting as ${profile.name || "Guest"} · ${profile.domain || "no domain"} · ${profile.expertise}`
                : userId
                  ? "Loading profile..."
                  : "Select a persona"}
            </p>
          </div>
        </header>

        <MessageList messages={messages} pending={pending} />

        <Composer
          disabled={!userId}
          pending={pending}
          model={model}
          onModelChange={setModel}
          onSend={send}
        />
      </main>
    </div>
  );
}
