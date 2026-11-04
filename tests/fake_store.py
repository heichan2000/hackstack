"""An in-memory ``Store`` for offline tests, registered as store ``memory``.

It honours the Store contract that matters to callers: upsert replaces a document's chunks,
and both retrievers filter by ACL principals and fail closed.
"""

from __future__ import annotations

import math
import re
from typing import Any

from hackstack.models import Chunk, RawDoc, SearchHit
from hackstack.registry import STORE, register

_WORD = re.compile(r"\w+")


@register(STORE, "memory")
class InMemoryStore:
    def __init__(self) -> None:
        self.docs: dict[str, RawDoc] = {}
        self.chunks: dict[str, Chunk] = {}

    def upsert(self, doc: RawDoc, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"chunk {c.chunk_id} has no embedding")
        self.delete_doc(doc.doc_id)
        self.docs[doc.doc_id] = doc
        for c in chunks:
            self.chunks[c.chunk_id] = c

    def delete_doc(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)
        for cid in [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]:
            del self.chunks[cid]

    def _visible(self, principals: list[str], filters: dict[str, Any] | None) -> list[Chunk]:
        allowed = set(principals)
        source = (filters or {}).get("source")
        return [
            c
            for c in self.chunks.values()
            if allowed & set(c.acl_principals) and (source is None or c.source == source)
        ]

    def dense(self, qvec, principals, filters, k) -> list[SearchHit]:
        scored = [(_cos(qvec, c.embedding or []), c) for c in self._visible(principals, filters)]
        return self._hits(scored, k, "dense")

    def lexical(self, query, principals, filters, k) -> list[SearchHit]:
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for c in self._visible(principals, filters):
            overlap = len(terms & set(_WORD.findall(c.embed_text.lower())))
            if overlap:
                scored.append((float(overlap), c))
        return self._hits(scored, k, "lexical")

    def _hits(self, scored, k, retriever) -> list[SearchHit]:
        scored.sort(key=lambda sc: sc[0], reverse=True)
        return [
            SearchHit(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                score=s,
                title=self.docs[c.doc_id].title,
                source=c.source,
                section_path=c.section_path,
                retriever=retriever,
            )
            for s, c in scored[:k]
        ]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
