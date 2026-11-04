"use client";

import { CheckIcon, FileTextIcon, LoaderCircleIcon, SearchIcon, TriangleAlertIcon } from "lucide-react";
import { useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ToolCall, Turn } from "@/lib/turn";
import type { Hit } from "@/lib/types";
import { cn } from "@/lib/utils";

const TOOL_LABEL: Record<string, string> = {
  search_kb: "Searching",
  retrieve: "Retrieving",
  get_document: "Reading",
  find_experts: "Finding experts",
  whats_changed: "Checking changes",
};

// The live panel: every tool call of the selected turn as it happens (query, hits, scores),
// then the passages the answer was grounded on. Clicking a hit or a [n] chip selects it.
export function SourcesPanel({
  turn,
  selected,
  onSelect,
}: {
  turn: Turn | null;
  selected: number | null;
  onSelect: (n: number) => void;
}) {
  if (!turn) {
    return (
      <Empty>Ask a question. Each search and the passages it found show up here as they stream.</Empty>
    );
  }
  const citedNumbers = new Set(Object.keys(turn.citations).map(Number));
  const cited = turn.sources.filter((s) => s.cited || (s.number != null && citedNumbers.has(s.number)));

  return (
    <ScrollArea className="h-full">
      <div className="space-y-5 p-4">
        <SelectedSource turn={turn} selected={selected} />

        <section>
          <Heading>Tool activity</Heading>
          {turn.tools.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {turn.status === "streaming" ? "Thinking…" : "No tools were called for this answer."}
            </p>
          ) : (
            <ol className="space-y-3">
              {turn.tools.map((call) => (
                <ToolCard key={call.id} call={call} selected={selected} onSelect={onSelect} />
              ))}
            </ol>
          )}
        </section>

        {turn.status === "done" && (
          <section>
            <Heading>Cited in this answer</Heading>
            {cited.length === 0 ? (
              <p className="text-sm text-muted-foreground">Nothing was cited.</p>
            ) : (
              <ul className="space-y-1">
                {cited.map((s) => (
                  <HitRow key={s.chunk_id} hit={s} selected={selected} onSelect={onSelect} />
                ))}
              </ul>
            )}
          </section>
        )}
      </div>
    </ScrollArea>
  );
}

function ToolCard({
  call,
  selected,
  onSelect,
}: {
  call: ToolCall;
  selected: number | null;
  onSelect: (n: number) => void;
}) {
  const label = TOOL_LABEL[call.name] ?? call.name;
  return (
    <li className="rounded-lg border bg-card p-3">
      <div className="flex items-start gap-2 text-sm">
        <span className="mt-0.5 text-muted-foreground">
          {call.status === "running" ? (
            <LoaderCircleIcon className="size-4 animate-spin" />
          ) : call.status === "error" ? (
            <TriangleAlertIcon className="size-4 text-destructive" />
          ) : call.name === "get_document" ? (
            <FileTextIcon className="size-4" />
          ) : call.name === "search_kb" || call.name === "retrieve" ? (
            <SearchIcon className="size-4" />
          ) : (
            <CheckIcon className="size-4" />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <div className="font-medium">
            {label}
            {call.query && <span className="font-normal text-muted-foreground"> “{call.query}”</span>}
            {call.status === "running" && "…"}
          </div>
          {call.status !== "running" && (
            <div className="text-xs text-muted-foreground">
              {call.status === "error"
                ? call.summary
                : `${call.hits.length} passage${call.hits.length === 1 ? "" : "s"}`}
              {call.seconds != null && ` · ${call.seconds.toFixed(1)} s`}
            </div>
          )}
        </div>
      </div>
      {call.hits.length > 0 && (
        <ul className="mt-2 space-y-1">
          {call.hits.map((h) => (
            <HitRow
              key={h.chunk_id}
              hit={h}
              range={scoreRange(call.hits)}
              selected={selected}
              onSelect={onSelect}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function HitRow({
  hit,
  range,
  selected,
  onSelect,
}: {
  hit: Hit;
  range?: [number, number];
  selected: number | null;
  onSelect: (n: number) => void;
}) {
  const active = hit.number != null && hit.number === selected;
  return (
    <li>
      <button
        type="button"
        disabled={hit.number == null}
        onClick={() => hit.number != null && onSelect(hit.number)}
        className={cn(
          "flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-xs transition-colors",
          active ? "bg-primary/10 ring-1 ring-primary/40" : "hover:bg-muted",
        )}
      >
        <span className="inline-flex h-5 min-w-5 items-center justify-center rounded border bg-muted px-1 font-mono text-[11px] font-semibold">
          {hit.number ?? "–"}
        </span>
        <span className="min-w-0 flex-1 truncate" title={hit.title}>
          {hit.title}
        </span>
        {hit.score != null && range && <ScoreBar score={hit.score} range={range} />}
      </button>
    </li>
  );
}

function scoreRange(hits: Hit[]): [number, number] {
  const scores = hits.map((h) => h.score).filter((s): s is number => s != null);
  return [Math.min(...scores), Math.max(...scores)];
}

// Scores aren't on one scale (reranker logits are often all negative; RRF is tiny), so the bar is
// relative to the other hits of the same call and the raw score is printed beside it.
function ScoreBar({ score, range: [lo, hi] }: { score: number; range: [number, number] }) {
  const width = hi > lo ? Math.round(15 + (85 * (score - lo)) / (hi - lo)) : 100;
  return (
    <span className="flex shrink-0 items-center gap-1.5 font-mono text-[10px] text-muted-foreground tabular-nums">
      <span className="h-1.5 w-10 overflow-hidden rounded-full bg-muted">
        <span className="block h-full rounded-full bg-primary/70" style={{ width: `${width}%` }} />
      </span>
      {score.toFixed(2)}
    </span>
  );
}

function SelectedSource({ turn, selected }: { turn: Turn; selected: number | null }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);
  if (selected == null) return null;
  const citation = turn.citations[selected];
  const hit =
    turn.sources.find((s) => s.number === selected) ??
    turn.tools.flatMap((c) => c.hits).find((h) => h.number === selected);
  if (!citation && !hit) return null;
  const title = citation?.title ?? hit?.title ?? "";
  const url = citation?.url ?? hit?.url;
  return (
    <div ref={ref} className="rounded-lg border border-primary/40 bg-primary/5 p-3">
      <div className="mb-1 flex items-center gap-2">
        <Badge>{selected}</Badge>
        <span className="truncate text-sm font-medium" title={title}>
          {title}
        </span>
      </div>
      <div className="mb-2 font-mono text-[11px] text-muted-foreground">
        {citation?.doc_id ?? hit?.doc_id}
      </div>
      {citation?.cited_text ? (
        <blockquote className="max-h-72 overflow-y-auto border-l-2 border-primary/50 pl-3 text-sm whitespace-pre-wrap text-foreground/90">
          {citation.cited_text}
        </blockquote>
      ) : (
        <p className="text-xs text-muted-foreground">Shown to the model; not quoted in this answer.</p>
      )}
      {url && (
        <a href={url} target="_blank" rel="noreferrer" className="mt-2 block text-xs text-primary underline">
          Open source
        </a>
      )}
    </div>
  );
}

function Heading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">{children}</h3>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="p-6 text-sm text-muted-foreground">{children}</div>;
}
