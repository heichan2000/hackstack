// One assistant turn, folded from the /chat event stream. Pure, so it is unit-tested.
import type { ChatEvent, Citation, Hit, Source } from "./types";

export type ToolCall = {
  id: string;
  name: string;
  query: string | null;
  status: "running" | "done" | "error";
  hits: Hit[];
  seconds: number | null;
  summary: string;
};

export type Turn = {
  question: string;
  status: "streaming" | "done" | "error";
  preamble: string;
  answer: string;
  tools: ToolCall[];
  citations: Record<number, Citation>;
  sources: Source[];
  lowGrounding: boolean | null;
  stopReason: string | null;
  seconds: number | null;
  mode: string | null;
  error: string | null;
};

export function emptyTurn(question: string): Turn {
  return {
    question,
    status: "streaming",
    preamble: "",
    answer: "",
    tools: [],
    citations: {},
    sources: [],
    lowGrounding: null,
    stopReason: null,
    seconds: null,
    mode: null,
    error: null,
  };
}

export function reduceTurn(t: Turn, ev: ChatEvent): Turn {
  switch (ev.type) {
    case "text_delta":
      return { ...t, answer: t.answer + ev.text };
    case "tool_start": {
      // Text streamed before a tool call was the model thinking aloud; the answer comes after.
      const preamble = [t.preamble, t.answer].filter(Boolean).join(" ");
      const call: ToolCall = {
        id: ev.id,
        name: ev.name,
        query: ev.query,
        status: "running",
        hits: [],
        seconds: null,
        summary: "",
      };
      return { ...t, preamble, answer: "", tools: [...t.tools, call] };
    }
    case "tool_end":
      return {
        ...t,
        tools: t.tools.map((c) =>
          c.id === ev.id
            ? {
                ...c,
                status: ev.is_error ? "error" : "done",
                hits: ev.hits ?? [],
                seconds: ev.seconds,
                summary: ev.summary,
              }
            : c,
        ),
      };
    case "citation":
      if (ev.number == null) return t;
      return { ...t, citations: { ...t.citations, [ev.number]: ev.citation } };
    case "grounding":
      return { ...t, sources: ev.sources, lowGrounding: ev.low_grounding };
    case "done":
      return {
        ...t,
        status: "done",
        answer: ev.answer,
        stopReason: ev.stop_reason,
        seconds: ev.seconds,
        mode: ev.mode,
        lowGrounding: t.lowGrounding ?? ev.low_grounding ?? null,
      };
    case "error":
      return { ...t, status: "error", error: ev.message };
    default:
      return t;
  }
}
