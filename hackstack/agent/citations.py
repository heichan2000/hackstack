"""Citations: the search_result blocks the model sees, and mapping its citations back.

Every passage a tool hands the model becomes one ``search_result`` block whose ``source`` is
``kb://<chunk_id>``. Claude echoes that source in its native ``search_result_location``
citations; the OpenAI-compatible adapter numbers the blocks and maps ``[n]`` markers back. Both
arrive as :class:`~hackstack.models.Citation` with the chunk id set.

:class:`CitationRegistry` is the per-conversation index of those blocks. Numbers count every
block across the whole conversation, in order (Claude's ``search_result_index`` and the local
adapter's ``[n]`` both count that way), so the registry is rebuilt from the stored history with
:meth:`CitationRegistry.from_messages` rather than kept on the side.

Rules that break naive implementations (research doc, *Generation and citations*):

- citations are all-or-nothing, and a ``tool_result`` holding any ``search_result`` must hold
  only ``search_result`` blocks, so an empty search returns the plain text "No results found.";
- granularity is per text block, so passages are split into paragraphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hackstack.llm.citations import KB_SCHEME, search_result_text
from hackstack.models import Citation, SearchHit

NO_RESULTS = "No results found."
MAX_BLOCKS = 6  # text blocks per passage: enough for sentence-ish citations, not a flood


@dataclass(frozen=True)
class SourceRef:
    """One passage given to the model: its number in the conversation and where it came from."""

    number: int
    chunk_id: str
    doc_id: str
    title: str  # human-readable: "Doc > Section (updated YYYY-MM-DD)"
    doc_title: str = ""
    url: str | None = None
    updated_at: datetime | None = None
    text: str = ""


def display_title(hit: SearchHit) -> str:
    """``Doc title > Section > Sub (updated 2026-03-02)``: what the model reads as the title.

    Human-readable titles rather than ids reduce hallucinated references (Anthropic, *Writing
    tools for agents*); the date lets the model reason about which of two sources is newer.
    """
    title = hit.title or hit.doc_id
    trail = list(hit.section_path)
    if trail and trail[0].strip().lower() == title.strip().lower():
        trail = trail[1:]
    text = " > ".join([title, *trail])
    if hit.updated_at is not None:
        text += f" (updated {hit.updated_at.date().isoformat()})"
    return text


def no_results_block(text: str = NO_RESULTS) -> dict[str, Any]:
    return {"type": "text", "text": text}


def is_low_grounding(citations: list[Citation]) -> bool:
    """An answer that cites nothing gets a "low grounding" badge."""
    return not citations


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()] or [text.strip() or "(empty)"]
    if len(parts) > MAX_BLOCKS:
        parts = [*parts[: MAX_BLOCKS - 1], "\n\n".join(parts[MAX_BLOCKS - 1 :])]
    return parts


class CitationRegistry:
    """Index -> chunk for one conversation. ``add`` a hit to get its ``search_result`` block."""

    def __init__(self) -> None:
        self.refs: list[SourceRef] = []
        self._by_chunk: dict[str, list[SourceRef]] = {}

    def add(self, hit: SearchHit) -> dict[str, Any]:
        """Register ``hit`` as the next numbered source and return its ``search_result`` block."""
        ref = SourceRef(
            number=len(self.refs) + 1,
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            title=display_title(hit),
            doc_title=hit.title,
            url=hit.url,
            updated_at=hit.updated_at,
            text=hit.text,
        )
        self._remember(ref)
        return {
            "type": "search_result",
            "source": f"{KB_SCHEME}{hit.chunk_id}",
            "title": ref.title,
            "content": [{"type": "text", "text": p} for p in _paragraphs(hit.text)],
            "citations": {"enabled": True},
        }

    def _remember(self, ref: SourceRef) -> None:
        self.refs.append(ref)
        self._by_chunk.setdefault(ref.chunk_id, []).append(ref)

    def seen(self, chunk_id: str) -> bool:
        return chunk_id in self._by_chunk

    def numbers(self, chunk_id: str) -> list[int]:
        return [r.number for r in self._by_chunk.get(chunk_id, [])]

    def ref(self, chunk_id: str) -> SourceRef | None:
        refs = self._by_chunk.get(chunk_id)
        return refs[0] if refs else None

    def resolve(self, citation: Citation) -> Citation:
        """``citation`` with the title and url of the source it points at (if registered)."""
        ref = self.ref(citation.chunk_id or "")
        if ref is None:
            return citation
        return citation.model_copy(
            update={"doc_id": ref.doc_id, "title": ref.title, "url": ref.url}
        )

    def find_doc(self, name: str) -> str | None:
        """The doc id of a registered source named by doc id, ``kb://`` source or title."""
        key = name.strip().removeprefix(KB_SCHEME)
        for ref in self.refs:
            if key in (ref.doc_id, ref.chunk_id):
                return ref.doc_id
        low = key.lower()
        for ref in self.refs:
            titles = {ref.title.lower(), ref.doc_title.lower()}
            if low in titles or (ref.doc_title and low.startswith(ref.doc_title.lower() + " >")):
                return ref.doc_id
        return None

    @classmethod
    def from_messages(cls, messages: list[dict[str, Any]]) -> CitationRegistry:
        """Rebuild the registry from a stored conversation, numbering blocks in order."""
        reg = cls()
        for message in messages:
            if message.get("role") != "user" or isinstance(message.get("content"), str):
                continue
            for block in message.get("content") or []:
                inner = block.get("content") if block.get("type") == "tool_result" else [block]
                if not isinstance(inner, list):
                    continue
                for b in inner:
                    if b.get("type") == "search_result":
                        reg._remember(_ref_from_block(b, len(reg.refs) + 1))
        return reg


def _ref_from_block(block: dict[str, Any], number: int) -> SourceRef:
    chunk_id = str(block.get("source", "")).removeprefix(KB_SCHEME)
    doc_id = chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else chunk_id
    title = block.get("title", "")
    return SourceRef(
        number=number,
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        doc_title=title.split(" > ")[0].split(" (updated ")[0],
        text=search_result_text(block),
    )
