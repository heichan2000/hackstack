import { describe, expect, it } from "vitest";
import { readEvents } from "./sse";

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch));
      c.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>) {
  const out: unknown[] = [];
  for await (const ev of readEvents(stream)) out.push(ev);
  return out;
}

describe("readEvents", () => {
  it("parses event/data frames into JSON events", async () => {
    const s = streamOf(
      'event: text_delta\ndata: {"type": "text_delta", "text": "Hi "}\n\n',
      'event: done\ndata: {"type": "done", "answer": "Hi"}\n\n',
    );
    expect(await collect(s)).toEqual([
      { type: "text_delta", text: "Hi " },
      { type: "done", answer: "Hi" },
    ]);
  });

  it("reassembles frames split across chunks, even mid-character", async () => {
    const frame = 'event: text_delta\ndata: {"type":"text_delta","text":"café"}\n\n';
    const bytes = new TextEncoder().encode(frame);
    const cut = bytes.indexOf(0xc3) + 1; // split inside the two-byte "é"
    const s = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, 5));
        c.enqueue(bytes.slice(5, cut));
        c.enqueue(bytes.slice(cut));
        c.close();
      },
    });
    expect(await collect(s)).toEqual([{ type: "text_delta", text: "café" }]);
  });

  it("accepts CRLF line endings and ignores comments and empty frames", async () => {
    const s = streamOf(': keep-alive\r\n\r\n', 'data: {"type":"grounding"}\r\n\r\n');
    expect(await collect(s)).toEqual([{ type: "grounding" }]);
  });

  it("joins multiple data lines with newlines", async () => {
    const s = streamOf('data: {"type":"x",\ndata: "v":1}\n\n');
    expect(await collect(s)).toEqual([{ type: "x", v: 1 }]);
  });

  it("flushes a final frame without a trailing blank line", async () => {
    const s = streamOf('data: {"type":"done"}');
    expect(await collect(s)).toEqual([{ type: "done" }]);
  });
});
