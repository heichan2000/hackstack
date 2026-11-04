// Split answer text into plain runs and [n] / [n, m] citation markers (the server's format:
// hackstack/llm/openai_compat_provider.py _MARKER).

export type Segment = { kind: "text"; text: string } | { kind: "cite"; numbers: number[] };

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

export function segment(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ kind: "text", text: text.slice(last, at) });
    out.push({ kind: "cite", numbers: m[1].split(",").map((n) => Number(n.trim())) });
    last = at + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}
