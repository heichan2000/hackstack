"""Incremental sync against the Compose Postgres (``make up``); marked ``pg``, skipped without it.

Each step uses a *new* ``PostgresStore`` and ``SyncManager``, so whatever carries over between
runs (content hashes, vectors, cursors) came out of the database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hackstack.chunking.markdown import MarkdownChunker
from hackstack.connectors.local_files import LocalFilesConnector
from hackstack.ingest import Ingestor
from hackstack.parsing.router import ParserRouter
from hackstack.store.postgres import PostgresStore
from hackstack.sync.manager import SyncManager
from tests.test_sync import HANDBOOK, Clock, CountingEmbedder, set_mtime, write_corpus

pytestmark = pytest.mark.pg


@pytest.fixture
def pg_env(pg_dsn, tmp_path):
    root = write_corpus(tmp_path / "corpus")
    for path in root.rglob("*"):
        set_mtime(path, datetime.now(UTC) - timedelta(days=1))
    embedder = CountingEmbedder()
    clock = Clock()
    PostgresStore(pg_dsn, embed_model=embedder.model, dim=embedder.dim).reset()
    opened: list[PostgresStore] = []

    def run(**kw):
        store = PostgresStore(pg_dsn, embed_model=embedder.model, dim=embedder.dim)
        opened.append(store)
        conn = LocalFilesConnector()
        conn.configure({"path": str(root)}, {})
        chunker = MarkdownChunker(size=20, max_size=40, min_size=1)
        ing = Ingestor(conn, ParserRouter(), chunker, embedder=embedder, store=store)
        return SyncManager(ing, now=clock).run(**kw), store

    yield root, embedder, clock, run
    for s in opened:
        s.close()


def test_incremental_sync_end_to_end_on_postgres(pg_env):
    root, embedder, clock, run = pg_env

    first, store = run()
    assert first.mode == "full" and first.docs == 3 and first.chunks_embedded > 0
    assert store.get_cursor("local_files") == {"since": clock.last.isoformat()}
    total = store.count_chunks()

    # unchanged data, read in full: nothing is embedded
    embedder.embedded.clear()
    again, store = run(full=True)
    assert again.unchanged == 3 and again.docs == 0 and embedder.embedded == []
    assert store.count_chunks() == total

    # one edited section: only that chunk is embedded; the rest reuse their stored vectors
    clock.tick()
    (root / "handbook.md").write_text(HANDBOOK.replace("thirty", "fourteen"), "utf-8")
    edited, store = run()
    assert edited.mode == "poll" and edited.docs == 1
    assert len(embedder.embedded) == 1 and "fourteen" in embedder.embedded[0]
    assert edited.chunks_reused >= 2
    hits = store.lexical("fourteen", ["group:everyone"], None, 5)
    assert [h.doc_id for h in hits] == ["local_files:handbook.md"]
    # the reused vectors are the stored ones: dense search still finds the untouched sections
    qvec = embedder.embed_query("Handbook > Travel\n\nBook economy for flights under six hours.")
    assert store.dense(qvec, ["group:everyone"], None, 1)[0].text.startswith("Book economy")

    # a deleted source doc disappears from the index
    clock.tick()
    (root / "launch.md").unlink()
    deleted, store = run()
    assert deleted.deleted == 1
    assert store.doc_hashes("local_files").keys() == {
        "local_files:handbook.md",
        "local_files:hr/salaries.md",
    }
    assert not store.lexical("Friday", ["group:everyone"], None, 5)

    # the cursor moved with every run and survives a fresh connection
    _, store = run()
    assert store.get_cursor("local_files") == {"since": clock.last.isoformat()}


def test_reset_clears_the_sync_cursors(pg_env, pg_dsn):
    root, embedder, clock, run = pg_env
    run()
    store = PostgresStore(pg_dsn, embed_model=embedder.model, dim=embedder.dim)
    try:
        store.reset()
        assert store.get_cursor("local_files") is None
    finally:
        store.close()
    report, _ = run()
    assert report.mode == "full" and report.docs == 3
