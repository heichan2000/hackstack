from __future__ import annotations

from typing import Protocol, runtime_checkable

from hackstack.models import Chunk, ParsedDoc, RawDoc


@runtime_checkable
class Chunker(Protocol):
    """Register implementations with ``@register("chunker", "<name>")``.

    Chunks must carry the document's ACL (``Chunk.for_doc`` does this) and set ``embed_text``.
    """

    def chunk(self, doc: ParsedDoc, raw: RawDoc) -> list[Chunk]: ...
