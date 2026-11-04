"""Structure-first Markdown chunking: headings, then recursive splits down to ~``size`` tokens.

1. Split on headings (:func:`~hackstack.parsing.sections.split_sections`). A heading with no
   body of its own makes no chunk; it lives on in its children's breadcrumbs.
2. Split each section's body into blocks: paragraphs, fenced code, tables. A table or code
   block that fits in ``max_size`` is never split; a bigger table is split by rows with its
   header repeated on every piece.
3. Pack blocks greedily up to ``size`` tokens. A block larger than ``size`` is split
   recursively on newlines, then sentences, then spaces. ``overlap`` tokens from the end of
   one chunk start the next one within a section (tables and code never overlap).
4. A chunk under ``min_size`` merges into an adjacent chunk of the same section or of its
   parent/child section, when the result still fits in ``size``.

``embed_text`` is ``"{title} > {section path}"`` followed by a blank line and the chunk text.
Chunk IDs are ``{doc_id}#{version}-{ordinal}``. ``version`` covers the algorithm and its size
knobs, so re-chunking with other settings never silently reuses old IDs (which would invalidate
a golden set without anyone noticing).

Token counts are an offline approximation: words and punctuation marks each count as one
token, which is close to BPE tokenizers for English prose. No tokenizer download is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from hackstack.models import Chunk, ParsedDoc, RawDoc
from hackstack.parsing.sections import split_sections
from hackstack.registry import CHUNKER, register

ALGORITHM_VERSION = "md1"

_TOKEN = re.compile(r"\w+|[^\w\s]")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_TABLE_ROW = re.compile(r"^\s*\|")
_SEPARATORS = ["\n", r"(?<=[.!?])\s+", r"\s+"]


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


@dataclass
class _Piece:
    path: list[str]
    text: str
    atomic: bool = False  # a table or code block: never overlapped into


@register(CHUNKER, "markdown")
class MarkdownChunker:
    def __init__(
        self, size: int = 400, overlap: int = 0, max_size: int = 512, min_size: int = 50
    ) -> None:
        if not 0 < size <= max_size:
            raise ValueError(f"chunk size must be in (0, max_size={max_size}], got {size}")
        if not 0 <= overlap < size:
            raise ValueError(f"overlap must be in [0, size), got {overlap}")
        self.size, self.overlap, self.max_size, self.min_size = size, overlap, max_size, min_size

    @classmethod
    def from_settings(cls, settings: Any) -> MarkdownChunker:
        c = settings.chunk
        return cls(
            size=c.size,
            overlap=c.overlap,
            max_size=c.max_size,
            min_size=c.min_size,
        )

    @property
    def version(self) -> str:
        return f"{ALGORITHM_VERSION}-s{self.size}-o{self.overlap}-m{self.min_size}-x{self.max_size}"

    def chunk(self, doc: ParsedDoc, raw: RawDoc) -> list[Chunk]:
        sections = split_sections(doc.markdown)
        pieces: list[_Piece] = []
        for section in sections:
            if section.text.strip():
                pieces.extend(self._section_pieces(section.path, section.text))
        pieces = self._merge_small(pieces)

        chunks = []
        for ordinal, piece in enumerate(pieces):
            c = Chunk.for_doc(
                raw,
                ordinal=ordinal,
                text=piece.text,
                section_path=piece.path,
                embed_text=f"{_breadcrumb(raw.title, piece.path)}\n\n{piece.text}",
            )
            chunks.append(
                c.model_copy(update={"chunk_id": f"{raw.doc_id}#{self.version}-{ordinal}"})
            )
        return chunks

    # --- within a section --------------------------------------------------------------------

    def _section_pieces(self, path: list[str], text: str) -> list[_Piece]:
        out: list[_Piece] = []
        buf: list[str] = []
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_tokens
            if buf:
                out.append(_Piece(path, "\n\n".join(buf)))
            buf, buf_tokens = [], 0

        for block, kind in _blocks(text):
            n = count_tokens(block)
            if kind in ("table", "code"):
                flush()
                if n <= self.max_size:
                    out.append(_Piece(path, block, atomic=True))
                else:
                    for part in self._split_atomic(block, kind):
                        out.append(_Piece(path, part, atomic=True))
                continue
            if n > self.size:
                flush()
                for part in _split_recursive(block, self.size, _SEPARATORS):
                    out.append(_Piece(path, part))
                continue
            if buf_tokens + n > self.size:
                flush()
            buf.append(block)
            buf_tokens += n
        flush()
        return self._add_overlap(out)

    def _split_atomic(self, block: str, kind: str) -> list[str]:
        lines = block.split("\n")
        if kind == "table" and len(lines) > 2:
            header, rows = lines[:2], lines[2:]
        elif kind == "code":
            header, rows = [], lines
        else:
            header, rows = [], lines
        budget = self.size - count_tokens("\n".join(header))
        parts, cur, cur_n = [], [], 0
        for row in rows:
            n = count_tokens(row)
            if cur and cur_n + n > budget:
                parts.append("\n".join(header + cur))
                cur, cur_n = [], 0
            cur.append(row)
            cur_n += n
        if cur:
            parts.append("\n".join(header + cur))
        return parts

    def _add_overlap(self, pieces: list[_Piece]) -> list[_Piece]:
        if not self.overlap:
            return pieces
        out = [pieces[0]] if pieces else []
        for prev, cur in zip(pieces, pieces[1:], strict=False):
            if prev.atomic or cur.atomic:
                out.append(cur)
                continue
            tail = _tail_tokens(prev.text, self.overlap)
            if tail and count_tokens(tail) + count_tokens(cur.text) <= self.max_size:
                cur = _Piece(cur.path, f"{tail} {cur.text}")
            out.append(cur)
        return out

    # --- across sections ---------------------------------------------------------------------

    def _merge_small(self, pieces: list[_Piece]) -> list[_Piece]:
        out: list[_Piece] = []
        for piece in pieces:
            if out and self._can_merge(out[-1], piece):
                out[-1] = _merge(out[-1], piece)
            else:
                out.append(piece)
        return out

    def _can_merge(self, a: _Piece, b: _Piece) -> bool:
        na, nb = count_tokens(a.text), count_tokens(b.text)
        if min(na, nb) >= self.min_size:
            return False
        extra = 0 if a.path == b.path else count_tokens(_heading_for(b.path))
        if na + nb + extra > self.size:
            return False
        common = _common_prefix(a.path, b.path)
        return bool(common) or common in (a.path, b.path)


def _merge(a: _Piece, b: _Piece) -> _Piece:
    path = _common_prefix(a.path, b.path)
    if a.path == b.path:
        return _Piece(path, f"{a.text}\n\n{b.text}")
    a_text = a.text if a.path == path else f"{_heading_for(a.path)}\n\n{a.text}"
    b_text = b.text if b.path == path else f"{_heading_for(b.path)}\n\n{b.text}"
    return _Piece(path, f"{a_text}\n\n{b_text}")


def _heading_for(path: list[str]) -> str:
    return f"{'#' * min(max(len(path), 1), 6)} {path[-1]}" if path else ""


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return out


def _breadcrumb(title: str, path: list[str]) -> str:
    trail = path[1:] if path and path[0].strip().lower() == title.strip().lower() else path
    return " > ".join([title, *trail]) if title else " > ".join(trail)


def _blocks(text: str) -> list[tuple[str, str]]:
    """Split a section body into (block, kind) with kind in paragraph | table | code."""
    blocks: list[tuple[str, str]] = []
    para: list[str] = []
    lines = text.split("\n")
    i = 0

    def flush_para() -> None:
        if para and "".join(para).strip():
            blocks.append(("\n".join(para).strip("\n"), "paragraph"))
        para.clear()

    while i < len(lines):
        line = lines[i]
        m = _FENCE.match(line)
        if m:
            flush_para()
            marker = m.group(1)[0]
            j = i + 1
            while j < len(lines) and not (
                (mm := _FENCE.match(lines[j])) and mm.group(1)[0] == marker
            ):
                j += 1
            blocks.append(("\n".join(lines[i : j + 1]), "code"))
            i = j + 1
            continue
        if _TABLE_ROW.match(line):
            flush_para()
            j = i
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                j += 1
            blocks.append(("\n".join(lines[i:j]), "table"))
            i = j
            continue
        if not line.strip():
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()
    return blocks


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    """Split ``text`` into pieces of at most ``size`` tokens, preferring coarse separators."""
    if count_tokens(text) <= size:
        return [text.strip()] if text.strip() else []
    if not separators:
        # no separator left (one enormous "word"): cut by token boundaries
        tokens = [m.group(0) for m in _TOKEN.finditer(text)]
        return ["".join(tokens[i : i + size]) for i in range(0, len(tokens), size)]
    sep, rest = separators[0], separators[1:]
    parts = [p for p in re.split(sep, text) if p.strip()]
    joiner = "\n" if sep == "\n" else " "
    out: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for part in parts:
        n = count_tokens(part)
        if n > size:
            if cur:
                out.append(joiner.join(cur))
                cur, cur_n = [], 0
            out.extend(_split_recursive(part, size, rest))
            continue
        if cur and cur_n + n > size:
            out.append(joiner.join(cur))
            cur, cur_n = [], 0
        cur.append(part.strip())
        cur_n += n
    if cur:
        out.append(joiner.join(cur))
    return out


def _tail_tokens(text: str, n: int) -> str:
    """The trailing whitespace-separated words of ``text`` totalling about ``n`` tokens."""
    words = text.split()
    tail: list[str] = []
    total = 0
    for word in reversed(words):
        t = count_tokens(word)
        if total + t > n:
            break
        tail.append(word)
        total += t
    return " ".join(reversed(tail))
