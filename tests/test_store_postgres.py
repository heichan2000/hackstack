"""Integration tests for the Postgres store, run against the Compose Postgres (``make up``).

They are marked ``pg`` and skip cleanly when Postgres is not reachable, so ``make test`` stays
offline-safe. Point them elsewhere with ``TEST_PG_DSN``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from hackstack.models import ACL, Chunk, RawDoc
from hackstack.store.postgres import IndexMismatchError, PostgresStore, vector_literal

pytestmark = pytest.mark.pg

DIM = 4
MODEL = "test-embed"
T0 = datetime(2026, 11, 1, tzinfo=UTC)


def doc(doc_id, principals, *, source="local_files", updated_at=T0, title=None):
    return RawDoc(
        doc_id=doc_id,
        source=source,
        title=title or doc_id,
        content=f"content of {doc_id}",
        url=f"file:///{doc_id}",
        updated_at=updated_at,
        acl=ACL(principals=principals),
    )


def chunks_for(raw, *texts_and_vecs):
    out = []
    for i, (text, vec) in enumerate(texts_and_vecs):
        c = Chunk.for_doc(raw, ordinal=i, text=text, section_path=["Top", f"s{i}"])
        c.embedding = vec
        out.append(c)
    return out


@pytest.fixture
def store(pg_dsn):
    s = PostgresStore(pg_dsn, embed_model=MODEL, dim=DIM)
    s.reset()
    yield s
    s.close()


def put(store, doc_id, principals, text, vec, **kw):
    raw = doc(doc_id, principals, **kw)
    store.upsert(raw, chunks_for(raw, (text, vec)))
    return raw


def ids(hits):
    return [h.chunk_id for h in hits]


# --- permissions ------------------------------------------------------------------------------


def test_a_principal_never_sees_another_principals_chunk_in_either_retriever(store):
    put(store, "alice-notes", ["user:alice@x.test"], "quarterly roadmap for alice", [1, 0, 0, 0])
    put(store, "eng-plan", ["group:eng"], "quarterly roadmap for engineering", [0.9, 0.1, 0, 0])
    put(store, "handbook", ["public"], "office hours and holidays", [0, 0, 1, 0])

    alice = ["user:alice@x.test", "public"]
    bob_eng = ["user:bob@x.test", "group:eng", "public"]
    q = [1, 0, 0, 0]

    assert set(ids(store.dense(q, alice, None, 10))) == {"alice-notes#0", "handbook#0"}
    assert ids(store.lexical("quarterly roadmap", alice, None, 10)) == ["alice-notes#0"]

    assert set(ids(store.dense(q, bob_eng, None, 10))) == {"eng-plan#0", "handbook#0"}
    assert ids(store.lexical("quarterly roadmap", bob_eng, None, 10)) == ["eng-plan#0"]


def test_no_principals_sees_nothing(store):
    put(store, "handbook", ["public"], "office hours", [1, 0, 0, 0])
    assert store.dense([1, 0, 0, 0], [], None, 10) == []
    assert store.lexical("office hours", [], None, 10) == []


def test_a_document_without_an_acl_is_private_to_the_ingesting_user(pg_dsn, store):
    owned = PostgresStore(pg_dsn, embed_model=MODEL, dim=DIM, ingest_user="ingest@x.test")
    try:
        put(owned, "scratch", [], "secret scratch notes", [1, 0, 0, 0])
    finally:
        owned.close()

    assert ids(store.lexical("scratch notes", ["user:ingest@x.test"], None, 5)) == ["scratch#0"]
    assert store.lexical("scratch notes", ["user:other@x.test", "public"], None, 5) == []
    assert store.dense([1, 0, 0, 0], ["user:other@x.test", "public"], None, 5) == []


def test_without_an_ingesting_user_an_acl_less_document_is_visible_to_nobody(store):
    put(store, "orphan", [], "orphan notes", [1, 0, 0, 0])
    assert store.lexical("orphan notes", ["public", "user:anyone@x.test"], None, 5) == []
    assert store.dense([1, 0, 0, 0], ["public", "user:anyone@x.test"], None, 5) == []


def test_a_restrictive_acl_does_not_shrink_top_k(pg_dsn, store):
    """pgvector filters after the index scan; iterative scans must keep returning rows."""

    def vec(i):
        return [1, (i % 7) / 10, (i % 5) / 10, (i % 3) / 100]

    # 20 of 300 chunks are visible to group:few
    few, many = doc("few", ["group:few"]), doc("many", ["group:many"])
    store.upsert(few, chunks_for(few, *[(f"note {i}", vec(i)) for i in range(0, 300, 15)]))
    store.upsert(many, chunks_for(many, *[(f"note {i}", vec(i)) for i in range(300) if i % 15]))
    store.analyze()
    assert store.count_chunks() == 300

    # At this size the planner would skip HNSW; force the index path so the test can fail.
    q = [1, 0.1, 0.1, 0]
    sql = (
        "SELECT chunk_id FROM chunks c WHERE c.acl_principals && %s::text[]"
        " ORDER BY c.embedding <=> %s::vector LIMIT 20"
    )
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        db = conn.execute("SELECT current_database()").fetchone()[0]
        conn.execute(f"ALTER DATABASE {db} SET enable_seqscan = off")
        conn.execute(f"ALTER DATABASE {db} SET enable_bitmapscan = off")
        try:
            with psycopg.connect(pg_dsn) as plain:  # control: pgvector defaults shrink top-k
                plain.execute("SET hnsw.ef_search = 40")
                plain.execute("SET hnsw.iterative_scan = off")
                shrunk = plain.execute(sql, (["group:few"], vector_literal(q))).fetchall()
            assert len(shrunk) < 20

            forced = PostgresStore(pg_dsn, embed_model=MODEL, dim=DIM)
            try:
                hits = forced.dense(q, ["group:few"], None, 20)
            finally:
                forced.close()
        finally:
            conn.execute(f"ALTER DATABASE {db} RESET ALL")
    assert len(hits) == 20
    assert {h.doc_id for h in hits} == {"few"}


# --- filters ----------------------------------------------------------------------------------


def test_source_filter_applies_to_both_retrievers(store):
    put(store, "s1", ["public"], "deploy runbook", [1, 0, 0, 0], source="notion")
    put(store, "s2", ["public"], "deploy runbook", [1, 0, 0, 0], source="slack")
    f = {"source": "slack"}
    assert ids(store.dense([1, 0, 0, 0], ["public"], f, 10)) == ["s2#0"]
    assert ids(store.lexical("deploy runbook", ["public"], f, 10)) == ["s2#0"]

    both = {"source": ["slack", "notion"]}
    assert len(store.dense([1, 0, 0, 0], ["public"], both, 10)) == 2
    assert len(store.lexical("deploy runbook", ["public"], both, 10)) == 2


def test_since_filter_applies_to_both_retrievers(store):
    put(store, "old", ["public"], "pricing change", [1, 0, 0, 0], updated_at=T0 - timedelta(days=9))
    put(store, "new", ["public"], "pricing change", [1, 0, 0, 0], updated_at=T0)
    f = {"since": T0 - timedelta(days=1)}
    assert ids(store.dense([1, 0, 0, 0], ["public"], f, 10)) == ["new#0"]
    assert ids(store.lexical("pricing change", ["public"], f, 10)) == ["new#0"]
    # ISO strings work too, identically on both sides
    f = {"since": (T0 - timedelta(days=1)).isoformat()}
    assert ids(store.dense([1, 0, 0, 0], ["public"], f, 10)) == ["new#0"]
    assert ids(store.lexical("pricing change", ["public"], f, 10)) == ["new#0"]


# --- embedding model / dimension guard -------------------------------------------------------


def test_opening_an_index_built_with_another_dimension_raises(pg_dsn, store):
    with pytest.raises(IndexMismatchError, match="make reset"):
        PostgresStore(pg_dsn, embed_model=MODEL, dim=DIM * 2).ensure_schema()


def test_opening_an_index_built_with_another_model_raises(pg_dsn, store):
    with pytest.raises(IndexMismatchError, match="other-model"):
        PostgresStore(pg_dsn, embed_model="other-model", dim=DIM).ensure_schema()


def test_upserting_a_vector_of_the_wrong_dimension_raises(store):
    raw = doc("bad", ["public"])
    with pytest.raises(IndexMismatchError, match="dim"):
        store.upsert(raw, chunks_for(raw, ("text", [1.0, 0.0])))


def test_querying_with_a_vector_of_the_wrong_dimension_raises(store):
    with pytest.raises(IndexMismatchError, match="dim"):
        store.dense([1.0, 0.0], ["public"], None, 5)


def test_a_chunk_without_an_embedding_is_refused(store):
    raw = doc("noembed", ["public"])
    chunk = Chunk.for_doc(raw, ordinal=0, text="no vector")
    with pytest.raises(ValueError, match="embedding"):
        store.upsert(raw, [chunk])


def test_reset_lets_a_new_embedder_take_over(pg_dsn, store):
    put(store, "a", ["public"], "alpha", [1, 0, 0, 0])
    bigger = PostgresStore(pg_dsn, embed_model="new-model", dim=8)
    try:
        bigger.reset()
        assert bigger.dense([1, 0, 0, 0, 0, 0, 0, 0], ["public"], None, 5) == []
    finally:
        bigger.close()


# --- upsert / delete -------------------------------------------------------------------------


def test_upsert_is_idempotent_on_chunk_id(store):
    raw = doc("handbook", ["public"])
    chunks = chunks_for(raw, ("pto policy", [1, 0, 0, 0]), ("expense policy", [0, 1, 0, 0]))
    store.upsert(raw, chunks)
    store.upsert(raw, chunks)
    assert store.count_chunks() == 2

    chunks[0].text = chunks[0].embed_text = "pto carryover policy"
    store.upsert(raw, chunks)
    assert store.count_chunks() == 2
    assert ids(store.lexical("carryover", ["public"], None, 5)) == ["handbook#0"]


def test_reupserting_a_shorter_document_drops_its_stale_chunks(store):
    raw = doc("handbook", ["public"])
    store.upsert(raw, chunks_for(raw, ("one", [1, 0, 0, 0]), ("two", [0, 1, 0, 0])))
    store.upsert(raw, chunks_for(raw, ("one", [1, 0, 0, 0])))
    assert store.count_chunks() == 1
    assert store.lexical("two", ["public"], None, 5) == []


def test_an_acl_change_takes_effect_on_reupsert(store):
    put(store, "memo", ["group:eng"], "reorg memo", [1, 0, 0, 0])
    put(store, "memo", ["group:execs"], "reorg memo", [1, 0, 0, 0])
    assert store.lexical("reorg memo", ["group:eng"], None, 5) == []
    assert store.dense([1, 0, 0, 0], ["group:eng"], None, 5) == []
    assert ids(store.lexical("reorg memo", ["group:execs"], None, 5)) == ["memo#0"]


def test_delete_doc_removes_it_from_both_retrievers(store):
    put(store, "gone", ["public"], "temporary page", [1, 0, 0, 0])
    store.delete_doc("gone")
    assert store.count_chunks() == 0
    assert store.dense([1, 0, 0, 0], ["public"], None, 5) == []
    assert store.lexical("temporary page", ["public"], None, 5) == []


# --- hit shape --------------------------------------------------------------------------------


def test_hits_carry_document_metadata_and_retriever_name(store):
    put(store, "handbook", ["public"], "office hours", [1, 0, 0, 0], title="Employee Handbook")
    (d,) = store.dense([1, 0, 0, 0], ["public"], None, 5)
    (lx,) = store.lexical("office hours", ["public"], None, 5)
    for hit, name in ((d, "dense"), (lx, "lexical")):
        assert hit.retriever == name
        assert hit.doc_id == "handbook"
        assert hit.title == "Employee Handbook"
        assert hit.url == "file:///handbook"
        assert hit.source == "local_files"
        assert hit.section_path == ["Top", "s0"]
        assert hit.updated_at == T0
        assert hit.text == "office hours"
    assert d.score == pytest.approx(1.0)
    assert lx.score > 0


def test_dense_ranks_nearest_first(store):
    put(store, "far", ["public"], "far", [0, 1, 0, 0])
    put(store, "near", ["public"], "near", [1, 0.1, 0, 0])
    assert ids(store.dense([1, 0, 0, 0], ["public"], None, 5)) == ["near#0", "far#0"]


def test_lexical_matches_any_query_term_and_ranks_more_matches_first(store):
    put(store, "both", ["public"], "Project Falcon launch checklist", [1, 0, 0, 0])
    put(store, "one", ["public"], "launch party snacks", [0, 1, 0, 0])
    assert ids(store.lexical("when is the falcon launch?", ["public"], None, 5)) == [
        "both#0",
        "one#0",
    ]


def test_lexical_with_only_stopwords_or_blank_returns_nothing(store):
    put(store, "a", ["public"], "the and of", [1, 0, 0, 0])
    assert store.lexical("the of", ["public"], None, 5) == []
    assert store.lexical("   ", ["public"], None, 5) == []


def test_postgres_store_is_registered():
    from hackstack import registry

    assert registry.get(registry.STORE, "postgres") is PostgresStore
