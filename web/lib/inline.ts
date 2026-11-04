// Just enough inline Markdown for model answers: **bold**, `code`, [label](url). Block structure
// (lists, paragraphs) is left to `white-space: pre-wrap`. No HTML is ever interpreted.

export type Inline =
  | { kind: "text" | "bold" | "code"; text: string }
  | { kind: "link"; text: string; href: string };

const TOKEN = /\*\*([^*\n]+)\*\*|`([^`\n]+)`|\[([^\]\n]+)\]\(([^)\s]+)\)/g;

export function inline(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ kind: "text", text: text.slice(last, at) });
    if (m[1] !== undefined) out.push({ kind: "bold", text: m[1] });
    else if (m[2] !== undefined) out.push({ kind: "code", text: m[2] });
    else if (/^https?:\/\//.test(m[4])) out.push({ kind: "link", text: m[3], href: m[4] });
    else out.push({ kind: "text", text: m[3] });
    last = at + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}
