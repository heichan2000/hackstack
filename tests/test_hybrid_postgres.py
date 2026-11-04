"""Hybrid retrieval against the real Postgres store and the real Northwind corpus.

Marked ``pg``: skipped when the Compose Postgres is not reachable. The embedder is the
deterministic fake, so this checks the wiring and the permission guarantees, not answer quality
(the live-model check is ``hackstack search`` with the local profile).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hackstack.chunking.markdown import MarkdownChunker
from hackstack.config import Settings
from hackstack.connectors.local_files import LocalFilesConnector
from hackstack.ingest import Ingestor
from hackstack.models import ACL, Chunk, RawDoc
from hackstack.parsing import router as router_mod
from hackstack.parsing.router import ParserRouter
from hackstack.retrieval.acl import resolve_principals
from hackstack.retrieval.expand import Expander
from hackstack.retrieval.hybrid import HybridRetriever
from hackstack.retrieval.rerank import NoopReranker
from hackstack.store.postgres import PostgresStore
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.pg

REPO = Path(__file__).resolve().parents[1]
APPS = REPO / "apps"
CORPUS = APPS / "company_brain" / "sample_corpus"
PERSONAS = [
    "eng@northwind.test",
    "hr@northwind.test",
    "finance@northwind.test",
    "ceo@northwind.test",
]
QUERIES = [
    "PTO carryover",
    "compensation bands salary",
    "Q4 budget",
    "2027 pricing change",
    "on-call rotation",
    "checkout latency postmortem",
    "Friday deploys code review",
    "performance cycle",
]


@pytest.fixture(scope="module")
def northwind(pg_dsn):
    """The Northwind corpus ingested once into the throwaway test database."""
    emb = FakeEmbedder()
    store = PostgresStore(pg_dsn, embed_model=emb.model, dim=emb.dim)
    store.reset()
    conn = LocalFilesConnector()
    conn.configure({"path": str(CORPUS)}, {})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(router_mod, "docling_available", lambda: False)
        report = Ingestor(conn, ParserRouter(), MarkdownChunker(), emb, store).run()
    assert report.docs == 30 and not report.failures
    store.analyze()
    acls = {
        r["doc_id"]: set(r["acl_principals"])
        for r in store._ready_conn().execute("SELECT doc_id, acl_principals FROM documents")
    }
    yield store, emb, acls
    store.close()


def make(store, emb, **kw):
    return HybridRetriever(
        store=store,
        embedder=emb,
        reranker=NoopReranker(),
        expander=Expander(store, radius=1),
        **kw,
    )


@pytest.mark.parametrize("user", PERSONAS)
def test_no_stage_leaks_a_document_the_persona_cannot_see(northwind, user):
    store, emb, acls = northwind
    principals = resolve_principals(Settings(app="company_brain"), user, apps_dir=APPS)
    visible = {d for d, acl in acls.items() if acl & set(principals)}
    r = make(store, emb, top_n=10)
    for q in QUERIES:
        res = r.retrieve(q, principals)
        assert res.hits, q
        for stage in (res.dense, res.lexical, res.fused, res.reranked, res.hits):
            assert {h.doc_id for h in stage} <= visible, (user, q)
        neighbours = {n.rsplit("#", 1)[0] for h in res.hits for n in h.neighbor_ids}
        assert neighbours <= visible, (user, q)


def test_role_switch_changes_what_compensation_returns(northwind):
    store, emb, _ = northwind
    settings = Settings(app="company_brain")
    r = make(store, emb, top_n=5)
    q = "compensation bands 2026"

    def docs(user):
        return {h.doc_id for h in r.search(q, resolve_principals(settings, user, apps_dir=APPS))}

    comp = [d for d in docs("hr@northwind.test") if "compensation-bands" in d]
    assert comp, "HR should retrieve the compensation bands doc"
    assert not [d for d in docs("eng@northwind.test") if "compensation-bands" in d]


def test_a_query_returns_fused_reranked_expanded_hits(northwind):
    store, emb, _ = northwind
    principals = resolve_principals(
        Settings(app="company_brain"), "eng@northwind.test", apps_dir=APPS
    )
    res = make(store, emb, top_n=5).retrieve("PTO carryover", principals)
    assert res.dense and res.lexical
    assert all(h.retriever == "hybrid" for h in res.fused)
    assert any("pto" in h.doc_id for h in res.hits[:3])
    assert any(h.neighbor_ids for h in res.hits)  # expansion folded neighbours in
    for h in res.hits:
        assert h.ordinal is not None


def test_neighbors_returns_windows_in_order_under_the_acl(pg_dsn):
    store = PostgresStore(pg_dsn, embed_model="test-embed", dim=2)
    store.reset()
    try:
        raw = RawDoc(
            doc_id="d",
            source="local_files",
            title="D",
            content="",
            acl=ACL(principals=["group:eng"]),
        )
        chunks = []
        for i in range(5):
            c = Chunk.for_doc(raw, ordinal=i, text=f"t{i}")
            c.embedding = [1.0, 0.0]
            chunks.append(c)
        store.upsert(raw, chunks)
        got = store.neighbors([("d", 1, 2), ("d", 2, 3)], ["group:eng"])
        assert [h.chunk_id for h in got] == ["d#1", "d#2", "d#3"]
        assert [h.ordinal for h in got] == [1, 2, 3]
        assert all(h.retriever == "neighbor" and h.title == "D" for h in got)
        assert store.neighbors([("d", 0, 4)], ["group:hr"]) == []
        assert store.neighbors([("d", 0, 4)], []) == []
        assert store.neighbors([], ["group:eng"]) == []
        # retrievers now report each hit's ordinal too
        assert store.lexical("t3", ["group:eng"], None, 5)[0].ordinal == 3
    finally:
        store.close()
