"use client";

import {
  ArrowUpIcon,
  BotIcon,
  CircleAlertIcon,
  LoaderCircleIcon,
  RotateCcwIcon,
  SquarePenIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Answer } from "@/components/answer";
import { SourcesPanel } from "@/components/sources-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, API_URL, getUsers, newConversationId, streamChat } from "@/lib/api";
import { emptyTurn, reduceTurn, type Turn } from "@/lib/turn";
import type { Mode, User, UsersResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

const SUGGESTIONS: Record<string, string[]> = {
  company_brain: [
    "How many PTO days carry over into next year?",
    "What is the salary range for a senior engineer?",
    "Can we deploy on Fridays, and how many PR approvals do we need?",
  ],
};

export function Chat() {
  const [meta, setMeta] = useState<(UsersResponse & { app?: string }) | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("agent");
  // Created on the first question (crypto.randomUUID is client-only, so not during prerender).
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);
  const [selectedCite, setSelectedCite] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const busy = turns.at(-1)?.status === "streaming";

  useEffect(() => {
    const ac = new AbortController();
    getUsers(ac.signal)
      .then((m) => {
        setMeta(m);
        setUserId(m.default_user ?? m.users[0]?.id ?? null);
      })
      .catch((e) => {
        if (!ac.signal.aborted) setLoadError(`Can't reach the API at ${API_URL} (${e.message}). Is \`make dev\` running?`);
      });
    return () => ac.abort();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  const ask = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || !userId || busy) return;
      const cid = conversationId ?? newConversationId();
      if (!conversationId) setConversationId(cid);
      const index = turns.length;
      const update = (fn: (t: Turn) => Turn) =>
        setTurns((ts) => ts.map((t, i) => (i === index ? fn(t) : t)));
      setTurns((ts) => [...ts, emptyTurn(q)]);
      setSelectedTurn(index);
      setSelectedCite(null);
      setInput("");
      const ac = new AbortController();
      abortRef.current = ac;
      try {
        for await (const ev of streamChat(
          { conversation_id: cid, message: q, as_user: userId, mode },
          ac.signal,
        )) {
          update((t) => reduceTurn(t, ev));
        }
        update((t) =>
          t.status === "streaming" ? { ...t, status: "error", error: "The stream ended early." } : t,
        );
      } catch (e) {
        if (ac.signal.aborted) return;
        const message =
          e instanceof ApiError && e.status === 409
            ? "This conversation belongs to another user. Start a new chat."
            : e instanceof Error
              ? e.message
              : String(e);
        update((t) => ({ ...t, status: "error", error: message }));
      }
    },
    [busy, conversationId, mode, turns.length, userId],
  );

  const newChat = (keepQuestion?: string) => {
    abortRef.current?.abort();
    setConversationId(null);
    setTurns([]);
    setSelectedTurn(null);
    setSelectedCite(null);
    setInput(keepQuestion ?? "");
  };

  // A conversation belongs to the user who started it, so switching users starts a new one.
  // The last question is kept in the composer: "same question, different answer".
  const switchUser = (id: string) => {
    if (id === userId) return;
    const lastQuestion = turns.at(-1)?.question;
    setUserId(id);
    newChat(lastQuestion);
  };

  const retry = (index: number) => {
    const q = turns[index].question;
    setTurns((ts) => ts.slice(0, index));
    // Nothing from a failed turn is stored server-side, so re-asking is safe.
    setTimeout(() => ask(q), 0);
  };

  const user = meta?.users.find((u) => u.id === userId) ?? null;
  const panelTurn = selectedTurn != null ? (turns[selectedTurn] ?? null) : (turns.at(-1) ?? null);
  const suggestions = (meta?.app && SUGGESTIONS[meta.app]) || [];

  return (
    <div className="flex h-dvh flex-col bg-background">
      <header className="flex flex-wrap items-center gap-3 border-b px-4 py-2.5">
        <div className="flex items-center gap-2">
          <BotIcon className="size-5 text-primary" />
          <h1 className="font-semibold">{meta?.title ?? "Hackstack"}</h1>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <ModeToggle mode={mode} onChange={setMode} disabled={busy} />
          {meta && (
            <UserSwitcher users={meta.users} value={userId} onChange={switchUser} />
          )}
          <Button variant="outline" size="sm" onClick={() => newChat()} disabled={turns.length === 0}>
            <SquarePenIcon /> New chat
          </Button>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[minmax(0,1fr)_380px]">
        <main className="flex min-h-0 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto">
            <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
              {loadError && (
                <Notice tone="error">
                  <CircleAlertIcon className="size-4" /> {loadError}
                </Notice>
              )}
              {user && turns.length === 0 && (
                <div className="space-y-4 pt-10 text-center">
                  <p className="text-muted-foreground">
                    Asking as <span className="font-medium text-foreground">{user.name}</span>
                    {" · "}
                    {user.role}
                  </p>
                  <div className="flex flex-wrap justify-center gap-1">
                    {user.groups.map((g) => (
                      <Badge key={g} variant="secondary">
                        {g}
                      </Badge>
                    ))}
                  </div>
                  {suggestions.length > 0 && (
                    <div className="mx-auto flex max-w-xl flex-col gap-2 pt-4">
                      {suggestions.map((s) => (
                        <button
                          key={s}
                          type="button"
                          onClick={() => ask(s)}
                          className="rounded-lg border px-3 py-2 text-left text-sm transition-colors hover:bg-muted"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {turns.map((t, i) => (
                <TurnView
                  key={i}
                  turn={t}
                  active={panelTurn === t}
                  selectedCite={panelTurn === t ? selectedCite : null}
                  onSelect={() => setSelectedTurn(i)}
                  onCite={(n) => {
                    setSelectedTurn(i);
                    setSelectedCite(n);
                  }}
                  onRetry={() => retry(i)}
                />
              ))}
              <div ref={bottomRef} />
            </div>
          </div>
          <Composer
            value={input}
            onChange={setInput}
            onSubmit={() => ask(input)}
            disabled={!userId}
            busy={busy}
            placeholder={user ? `Ask as ${user.name}…` : "Connecting…"}
          />
        </main>

        <aside className="hidden min-h-0 border-l bg-muted/30 md:block">
          <SourcesPanel turn={panelTurn} selected={selectedCite} onSelect={setSelectedCite} />
        </aside>
      </div>
    </div>
  );
}

function TurnView({
  turn,
  active,
  selectedCite,
  onSelect,
  onCite,
  onRetry,
}: {
  turn: Turn;
  active: boolean;
  selectedCite: number | null;
  onSelect: () => void;
  onCite: (n: number) => void;
  onRetry: () => void;
}) {
  const running = turn.tools.filter((c) => c.status === "running");
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2 text-primary-foreground whitespace-pre-wrap">
          {turn.question}
        </div>
      </div>
      <div
        onClick={onSelect}
        className={cn(
          "rounded-xl border px-4 py-3 transition-colors",
          active ? "border-primary/30 bg-card shadow-sm" : "cursor-pointer bg-card/60 hover:bg-card",
        )}
      >
        {turn.preamble && <p className="mb-2 text-sm text-muted-foreground italic">{turn.preamble}</p>}
        {turn.status === "streaming" && !turn.answer && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderCircleIcon className="size-4 animate-spin" />
            {running.length > 0
              ? `Searching ${running.map((c) => (c.query ? `“${c.query}”` : c.name)).join(", ")}…`
              : turn.tools.length > 0
                ? "Writing the answer…"
                : "Thinking…"}
          </p>
        )}
        {turn.answer && <Answer text={turn.answer} selected={selectedCite} onCite={onCite} />}
        {turn.status === "error" && (
          <div className="mt-2 flex items-center gap-3 text-sm text-destructive">
            <CircleAlertIcon className="size-4 shrink-0" />
            <span className="flex-1">{turn.error}</span>
            <Button size="sm" variant="outline" onClick={onRetry}>
              <RotateCcwIcon /> Retry
            </Button>
          </div>
        )}
        {turn.status === "done" && (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {turn.lowGrounding && (
              <Badge variant="outline" className="border-amber-500/60 text-amber-700 dark:text-amber-400">
                <TriangleAlertIcon /> Low grounding: no sources cited
              </Badge>
            )}
            {turn.stopReason && !["end_turn", "stop_sequence"].includes(turn.stopReason) && (
              <Badge variant="destructive">{STOP_LABEL[turn.stopReason] ?? turn.stopReason}</Badge>
            )}
            <span>
              {turn.tools.length} tool call{turn.tools.length === 1 ? "" : "s"}
              {turn.seconds != null && ` · ${turn.seconds.toFixed(1)} s`}
              {turn.mode && ` · ${turn.mode === "single_shot" ? "single-shot" : "agent"}`}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

const STOP_LABEL: Record<string, string> = {
  max_tokens: "Cut off (max tokens)",
  refusal: "Refused",
  max_iterations: "Tool budget ran out",
  model_context_window_exceeded: "Conversation too long: start a new chat",
};

function ModeToggle({
  mode,
  onChange,
  disabled,
}: {
  mode: Mode;
  onChange: (m: Mode) => void;
  disabled: boolean;
}) {
  const options: [Mode, string][] = [
    ["agent", "Agent"],
    ["single_shot", "Single-shot"],
  ];
  return (
    <div role="radiogroup" aria-label="Mode" className="inline-flex rounded-lg border p-0.5">
      {options.map(([value, label]) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={mode === value}
          disabled={disabled}
          onClick={() => onChange(value)}
          className={cn(
            "rounded-md px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-50",
            mode === value ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function UserSwitcher({
  users,
  value,
  onChange,
}: {
  users: User[];
  value: string | null;
  onChange: (id: string) => void;
}) {
  const items = Object.fromEntries(users.map((u) => [u.id, `${u.name} · ${u.role}`]));
  return (
    <Select items={items} value={value} onValueChange={(v) => v && onChange(v as string)}>
      <SelectTrigger size="sm" aria-label="Ask as user" className="min-w-56">
        <SelectValue placeholder="Choose a user" />
      </SelectTrigger>
      <SelectContent>
        {users.map((u) => (
          <SelectItem key={u.id} value={u.id}>
            <span className="font-medium">{u.name}</span>
            <span className="text-muted-foreground">{u.role}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  busy,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  busy: boolean;
  placeholder: string;
}) {
  return (
    <form
      className="border-t p-3"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              onSubmit();
            }
          }}
          rows={1}
          placeholder={placeholder}
          disabled={disabled}
          className="max-h-40 min-h-10 resize-none"
          aria-label="Message"
        />
        <Button type="submit" size="icon-lg" disabled={disabled || busy || !value.trim()} aria-label="Send">
          {busy ? <LoaderCircleIcon className="animate-spin" /> : <ArrowUpIcon />}
        </Button>
      </div>
    </form>
  );
}

function Notice({ tone, children }: { tone: "error"; children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-lg border px-3 py-2 text-sm",
        tone === "error" && "border-destructive/40 bg-destructive/5 text-destructive",
      )}
    >
      {children}
    </div>
  );
}
