"""Mapping search-result blocks back to the kit's :class:`~hackstack.models.Citation`.

Convention: a ``search_result`` block's ``source`` is the chunk id (``<doc_id>#<ordinal>``),
optionally prefixed ``kb://``, e.g. ``kb://local_files:handbook/pto.md#2``. Both adapters rely
on it: Claude echoes ``source`` in its ``search_result_location`` citations, and the
OpenAI-compatible fallback numbers the blocks it sees.
"""

from __future__ import annotations

from typing import Any

from hackstack.models import Citation

KB_SCHEME = "kb://"


def citation_from_source(source: str, *, title: str = "", cited_text: str = "") -> Citation:
    chunk_id = source.removeprefix(KB_SCHEME)
    doc_id = chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else chunk_id
    return Citation(doc_id=doc_id, chunk_id=chunk_id, title=title, cited_text=cited_text)


def search_result_text(block: dict[str, Any]) -> str:
    """The text of a ``search_result`` block (its text content blocks, joined)."""
    return "\n".join(
        part.get("text", "") for part in block.get("content") or [] if part.get("type") == "text"
    )
