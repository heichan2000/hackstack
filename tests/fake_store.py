"""An in-memory ``Store`` for offline tests, registered as store ``memory``.

It honours the Store contract that matters to callers: upsert replaces a document's chunks,
and both retrievers filter by ACL principals and fail closed.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, date, datetime
from typing import Any

from hackstack.models import Chunk, RawDoc, SearchHit
from hackstack.registry import STORE, register

_WORD = re.compile(r"\w+")


@register(STORE, "memory")
class InMemoryStore:
    def __init__(self) -> None:
        self.docs: dict[str, RawDoc] = {}
        self.chunks: dict[str, Chunk] = {}
        self.hashes: dict[str, str | None] = {}
        self.cursors: dict[str, dict[str, Any]] = {}

    def upsert(self, doc: RawDoc, chunks: list[Chunk], *, content_hash: str | None = None) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"chunk {c.chunk_id} has no embedding")
        self.delete_doc(doc.doc_id)
        self.docs[doc.doc_id] = doc
        self.hashes[doc.doc_id] = content_hash
        for c in chunks:
            self.chunks[c.chunk_id] = c

    def delete_doc(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)
        self.hashes.pop(doc_id, None)
        for cid in [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]:
            del self.chunks[cid]

    # --- sync (hackstack.sync.manager.SyncStore) ----------------------------------------------

    def doc_hashes(self, source: str) -> dict[str, str | None]:
        return {d: self.hashes.get(d) for d, doc in self.docs.items() if doc.source == source}

    def get_cursor(self, connector: str) -> dict[str, Any] | None:
        return dict(self.cursors[connector]) if connector in self.cursors else None

    def set_cursor(self, connector: str, cursor: dict[str, Any]) -> None:
        self.cursors[connector] = dict(cursor)

    def chunk_embeddings(self, doc_id: str) -> list[tuple[str, list[float]]]:
        return [(c.embed_text, c.embedding) for c in self.chunks.values() if c.doc_id == doc_id]

    def _visible(self, principals: list[str], filters: dict[str, Any] | None) -> list[Chunk]:
        allowed = set(principals)
        source = (filters or {}).get("source")
        since = (filters or {}).get("since")
        since_dt = _as_utc(since) if since is not None else None
        return [
            c
            for c in self.chunks.values()
            if allowed & set(c.acl_principals)
            and (source is None or c.source == source)
            and (
                since_dt is None or (c.updated_at is not None and _as_utc(c.updated_at) >= since_dt)
            )
        ]

    # --- app-tool reads (find_experts, whats_changed) -------------------------------------------

    def doc_owners(self, doc_ids, principals) -> dict[str, list[str]]:
        allowed = set(principals)
        return {
            d: list(self.docs[d].owners)
            for d in doc_ids
            if d in self.docs and allowed & set(self.docs[d].acl.principals)
        }

    def changed_since(self, principals, since, *, source=None, limit=20) -> list[SearchHit]:
        first: dict[str, Chunk] = {}
        for c in self._visible(principals, {"since": since, "source": source}):
            if c.doc_id not in first or c.ordinal < first[c.doc_id].ordinal:
                first[c.doc_id] = c
        ordered = sorted(first.values(), key=lambda c: c.doc_id)
        ordered.sort(key=lambda c: _as_utc(c.updated_at), reverse=True)
        return self._hits([(0.0, c) for c in ordered[:limit]], limit, "changed")

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
                updated_at=c.updated_at,
                ordinal=c.ordinal,
                retriever=retriever,
            )
            for s, c in scored[:k]
        ]

    def neighbors(self, windows, principals) -> list[SearchHit]:
        """Chunks by position (context expansion), ACL-filtered like the retrievers."""
        visible = self._visible(principals, None)
        picked = [
            c
            for c in visible
            if any(c.doc_id == d and lo <= c.ordinal <= hi for d, lo, hi in windows)
        ]
        picked.sort(key=lambda c: (c.doc_id, c.ordinal))
        return self._hits([(0.0, c) for c in picked], len(picked), "neighbor")


def _as_utc(value: datetime | date | str | None) -> datetime:
    """Like the Postgres store's: dates are midnight UTC, naive datetimes are UTC."""
    if value is None:
        return datetime.min.replace(tzinfo=UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
