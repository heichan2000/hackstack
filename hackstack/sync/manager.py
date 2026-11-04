"""The sync manager: incremental ingest on top of the :class:`~hackstack.ingest.Ingestor`.

It owns the two content hashes and the per-connector cursors:

* a **doc hash** over the document's content *and* metadata (ACL, title, owners, dates...) plus
  the pipeline settings: an unchanged document is skipped before it is parsed;
* a **chunk hash** over each chunk's ``embed_text``: an edited document is re-chunked, but only
  the chunks whose text changed are re-embedded; the rest reuse their stored vectors.

Connectors never touch the database; the manager talks to a :class:`SyncStore`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, runtime_checkable

from hackstack.connectors.base import ConnectorFailure
from hackstack.ingest import Ingestor, IngestReport
from hackstack.models import Chunk, RawDoc
from hackstack.store.base import Store

#: Polls start this long before the saved cursor. File systems stamp mtimes from a coarse clock
#: (~16 ms on Windows), remote APIs index late, and clocks drift; re-reading a minute of
#: changes costs only a hash check, while missing one costs freshness until the next full run.
POLL_OVERLAP = timedelta(minutes=1)


@runtime_checkable
class SyncStore(Store, Protocol):
    """What the sync manager needs from a store, on top of :class:`Store`."""

    def upsert(self, doc: RawDoc, chunks: list[Chunk], *, content_hash: str | None = None) -> None:
        """As :meth:`Store.upsert`, recording ``content_hash`` as the document's hash."""
        ...

    def get_cursor(self, connector: str) -> dict[str, Any] | None:
        """The connector's saved cursor (``sync_state``), or ``None`` before its first run."""
        ...

    def set_cursor(self, connector: str, cursor: dict[str, Any]) -> None: ...

    def doc_hashes(self, source: str) -> dict[str, str | None]:
        """``doc_id -> content_hash`` for every stored document from ``source``."""
        ...

    def chunk_embeddings(self, doc_id: str) -> list[tuple[str, list[float]]]:
        """``(embed_text, embedding)`` for each stored chunk of ``doc_id``."""
        ...


class SyncReport(IngestReport):
    """``docs``/``chunks`` count what was written; the rest says what was skipped or reused."""

    mode: Literal["full", "poll"] = "full"
    unchanged: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    deleted: int = 0


def doc_hash(raw: RawDoc, *, pipeline: str = "", volatile: frozenset[str] = frozenset()) -> str:
    """Hash of everything that decides a document's chunks and their ACL.

    ``volatile`` names ``extra`` keys that change without the document changing (a file's
    mtime after a fresh checkout); they are left out. ``pipeline`` identifies the parse/chunk
    settings, so changing them re-processes every document.
    """
    content = raw.content.encode("utf-8") if isinstance(raw.content, str) else raw.content
    meta = raw.model_dump(mode="json", exclude={"content"})
    meta["extra"] = {k: v for k, v in meta["extra"].items() if k not in volatile}
    blob = json.dumps({"meta": meta, "pipeline": pipeline}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(hashlib.sha256(content).digest() + blob).hexdigest()


def chunk_hash(embed_text: str) -> str:
    """Hash of what gets embedded: equal hashes can share a vector (same embedder per index)."""
    return hashlib.sha256(embed_text.encode("utf-8")).hexdigest()


class SyncManager:
    def __init__(
        self,
        ingestor: Ingestor,
        *,
        pipeline: str = "",
        now: Callable[[], datetime] | None = None,
        poll_overlap: timedelta = POLL_OVERLAP,
    ) -> None:
        if not isinstance(ingestor.store, SyncStore):
            raise TypeError(f"{type(ingestor.store).__name__} does not support incremental sync")
        self.ingestor = ingestor
        self.store: SyncStore = ingestor.store
        self.connector = ingestor.connector
        self.pipeline = pipeline or str(getattr(ingestor.chunker, "version", ""))
        self.volatile = frozenset(getattr(self.connector, "volatile_extra", ()))
        self.now = now or (lambda: datetime.now(UTC))
        self.poll_overlap = poll_overlap

    def run(self, *, full: bool = False) -> SyncReport:
        """Bring the index up to date with the connector's source.

        With a saved cursor and a connector that can ``poll``, only the window since the last
        run is read; otherwise (or with ``full``) everything is, and unchanged documents are
        still skipped by hash. Deletions are reconciled against ``list_ids()`` (or, on a full
        run without it, against what ``load_all`` yielded). The cursor is saved only when the
        run finishes, so a crashed run is simply repeated.
        """
        started = time.perf_counter()
        name = self.connector.name
        until = self.now()
        since = None if full else _cursor_since(self.store.get_cursor(name))
        poll = getattr(self.connector, "poll", None)
        list_ids = getattr(self.connector, "list_ids", None)
        mode = "poll" if since is not None and callable(poll) else "full"
        report = SyncReport(connector=name, mode=mode)

        stored = self.store.doc_hashes(name)
        seen: set[str] = set()
        if mode == "poll":
            assert since is not None and callable(poll)
            items = poll(since - self.poll_overlap, until)
        else:
            items = self.connector.load_all()
        self._consume(items, stored, seen, report)

        live: set[str] | None = set(seen) if mode == "full" else None
        if callable(list_ids):
            live = set(list_ids())
            # In the source but neither indexed nor polled: renamed or copied with an old
            # mtime. Load them now rather than wait for a full run.
            missing = live - set(stored) - seen
            if missing:
                wanted = (i for i in self.connector.load_all() if _item_id(i) in missing)
                self._consume(wanted, stored, seen, report)
        if live is not None:
            failed = {f.item_id for f in report.failures}
            for doc_id in sorted(set(stored) - live - failed):
                self.store.delete_doc(doc_id)
                report.deleted += 1

        if (report.docs or report.deleted) and callable(getattr(self.store, "analyze", None)):
            self.store.analyze()  # type: ignore[attr-defined]
        self.store.set_cursor(name, {"since": until.isoformat()})
        report.seconds = round(time.perf_counter() - started, 3)
        return report

    def _consume(
        self,
        items: Iterable[RawDoc | ConnectorFailure],
        stored: dict[str, str | None],
        seen: set[str],
        report: SyncReport,
    ) -> None:
        for item in items:
            seen.add(_item_id(item))
            if isinstance(item, ConnectorFailure):
                report.failures.append(item)
            else:
                self._sync_doc(item, stored, report)

    def _sync_doc(self, raw: RawDoc, stored: dict[str, str | None], report: SyncReport) -> None:
        h = doc_hash(raw, pipeline=self.pipeline, volatile=self.volatile)
        if stored.get(raw.doc_id) == h:
            report.unchanged += 1
            return
        chunks = self.ingestor.prepare(raw)
        if isinstance(chunks, ConnectorFailure):
            report.failures.append(chunks)
            return
        known = (
            {chunk_hash(t): vec for t, vec in self.store.chunk_embeddings(raw.doc_id)}
            if raw.doc_id in stored
            else {}
        )
        fresh = []
        for chunk in chunks:
            vec = known.get(chunk_hash(chunk.embed_text))
            if vec is None:
                fresh.append(chunk)
            else:
                chunk.embedding = vec
        if fresh:
            self.ingestor.embed(fresh)
        report.chunks_embedded += len(fresh)
        report.chunks_reused += len(chunks) - len(fresh)
        self.store.upsert(raw, chunks, content_hash=h)
        report.docs += 1
        report.chunks += len(chunks)


def _item_id(item: RawDoc | ConnectorFailure) -> str:
    return item.item_id if isinstance(item, ConnectorFailure) else item.doc_id


def _cursor_since(cursor: dict[str, Any] | None) -> datetime | None:
    if not cursor or not cursor.get("since"):
        return None
    since = datetime.fromisoformat(cursor["since"])
    return since if since.tzinfo else since.replace(tzinfo=UTC)
