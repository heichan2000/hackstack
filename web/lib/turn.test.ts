import { describe, expect, it } from "vitest";
import { emptyTurn, reduceTurn, type Turn } from "./turn";
import type { ChatEvent } from "./types";

function run(events: ChatEvent[]): Turn {
  return events.reduce(reduceTurn, emptyTurn("q"));
}

const hit = (number: number, title = `Doc ${number}`) => ({
  number,
  title,
  score: 0.5,
  doc_id: `d${number}`,
  chunk_id: `d${number}#0`,
  url: null,
});

describe("reduceTurn", () => {
  it("appends text deltas to the answer", () => {
    const t = run([
      { type: "text_delta", text: "Five " },
      { type: "text_delta", text: "days [1]." },
    ]);
    expect(t.answer).toBe("Five days [1].");
    expect(t.status).toBe("streaming");
  });

  it("treats text before a tool_start as a preamble, not the answer", () => {
    const t = run([
      { type: "text_delta", text: "Let me check." },
      { type: "tool_start", id: "c0", name: "search_kb", query: "pto", input: {} },
      { type: "text_delta", text: "Five days." },
    ]);
    expect(t.preamble).toBe("Let me check.");
    expect(t.answer).toBe("Five days.");
  });

  it("tracks parallel tool calls from start to end by id", () => {
    const t = run([
      { type: "tool_start", id: "c0", name: "search_kb", query: "a", input: {} },
      { type: "tool_start", id: "c1", name: "search_kb", query: "b", input: {} },
      {
        type: "tool_end", id: "c0", name: "search_kb", query: "a", is_error: false,
        results: 1, seconds: 1.2, summary: "", hits: [hit(1)],
      },
    ]);
    expect(t.tools.map((x) => [x.id, x.status])).toEqual([
      ["c0", "done"],
      ["c1", "running"],
    ]);
    expect(t.tools[0].hits).toHaveLength(1);
    expect(t.tools[0].seconds).toBe(1.2);
  });

  it("records citations by number and grounding sources", () => {
    const t = run([
      {
        type: "citation", number: 2,
        citation: { doc_id: "d2", chunk_id: "d2#0", title: "Doc 2", url: null, cited_text: "x" },
      },
      {
        type: "grounding", low_grounding: false, cited: [2],
        sources: [{ ...hit(1), cited: false }, { ...hit(2), cited: true }],
      },
    ]);
    expect(t.citations[2].cited_text).toBe("x");
    expect(t.sources.map((s) => s.number)).toEqual([1, 2]);
    expect(t.lowGrounding).toBe(false);
  });

  it("ignores a citation with a null number", () => {
    const t = run([
      {
        type: "citation", number: null,
        citation: { doc_id: "d", chunk_id: "d#0", title: "D", url: null, cited_text: "" },
      },
    ]);
    expect(Object.keys(t.citations)).toEqual([]);
  });

  it("takes done.answer as the final text and marks the turn done", () => {
    const t = run([
      { type: "text_delta", text: "partial" },
      {
        type: "done", conversation_id: "c", answer: "Final [1].", stop_reason: "end_turn",
        truncated: false, refusal: null, seconds: 6.8, mode: "agent", low_grounding: true,
      },
    ]);
    expect(t.answer).toBe("Final [1].");
    expect(t.status).toBe("done");
    expect(t.seconds).toBe(6.8);
    expect(t.stopReason).toBe("end_turn");
  });

  it("marks the turn failed on error", () => {
    const t = run([{ type: "error", code: "internal_error", message: "boom" }]);
    expect(t.status).toBe("error");
    expect(t.error).toBe("boom");
  });

  it("ignores unknown event types", () => {
    const t = run([{ type: "mystery" } as unknown as ChatEvent]);
    expect(t).toEqual(emptyTurn("q"));
  });
});
