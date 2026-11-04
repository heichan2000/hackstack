// A hand-rolled SSE reader for a fetch() response body: no EventSource (it can't POST), no SDK.
// Frames are separated by a blank line; only `data:` lines matter, since every payload repeats
// its event name as `type` (docs/sse.md).

export async function* readEvents<T = unknown>(body: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder(); // stream: true keeps a multi-byte char split across chunks
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
      let i: number;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const ev = parseFrame(buf.slice(0, i));
        buf = buf.slice(i + 2);
        if (ev !== undefined) yield ev as T;
      }
    }
    const last = parseFrame(buf + decoder.decode());
    if (last !== undefined) yield last as T;
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): unknown {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(line.startsWith("data: ") ? 6 : 5))
    .join("\n");
  return data ? JSON.parse(data) : undefined;
}
