"""The hybrid pipeline offline: dense + lexical -> RRF -> rerank shortlist -> expand."""

from __future__ import annotations

import pytest

from hackstack.config import Settings
from hackstack.models import ACL, Chunk, RawDoc
from hackstack.retrieval.expand import Expander
from hackstack.retrieval.hybrid import HybridRetriever
from hackstack.retrieval.rerank import NoopReranker
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder

DIM = 256
ENG = ["user:eng@northwind.test", "group:everyone", "group:engineering", "public"]
HR = ["user:hr@northwind.test", "group:everyone", "group:hr", "public"]

DOCS = {
    "pto-policy": (
        ["group:everyone"],
        [
            "Northwind PTO policy overview for all employees.",
            "PTO carryover: up to five unused days carry over into the next year.",
            "Requests go through the HR portal two weeks ahead.",
        ],
    ),
    "comp-bands": (
        ["group:hr", "group:leadership"],
        [
            "Compensation bands for 2026 by level.",
            "PTO carryover is not paid out; salary bands L3 to L7.",
            "Band reviews happen every March.",
        ],
    ),
    "deploy-runbook": (
        ["group:engineering"],
        [
            "Deploy runbook: freeze on Fridays after 2pm.",
            "Rollback with the release toggle in Atlas.",
        ],
    ),
}


@pytest.fixture
def store():
    s = InMemoryStore()
    emb = FakeEmbedder(dim=DIM)
    for doc_id, (acl, texts) in DOCS.items():
        raw = RawDoc(
            doc_id=doc_id,
            source="local_files",
            title=doc_id,
            content="",
            acl=ACL(principals=acl),
        )
        chunks = [Chunk.for_doc(raw, ordinal=i, text=t) for i, t in enumerate(texts)]
        for c, v in zip(chunks, emb.embed_documents([c.embed_text for c in chunks]), strict=True):
            c.embedding = v
        s.upsert(raw, chunks)
    return s


class RecordingReranker(NoopReranker):
    def __init__(self):
        super().__init__()
        self.seen: list[int] = []

    def rerank(self, query, hits, top_n):
        self.seen.append(len(hits))
        out = super().rerank(query, hits, top_n)
        return [h.model_copy(update={"retriever": "rerank"}) for h in out]


class CountingEmbedder(FakeEmbedder):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.queries = 0

    def embed_query(self, text):
        self.queries += 1
        return super().embed_query(text)


def retriever(store, **kw):
    kw.setdefault("embedder", FakeEmbedder(dim=DIM))
    kw.setdefault("reranker", NoopReranker())
    return HybridRetriever(store=store, **kw)


def test_a_query_returns_fused_reranked_expanded_hits(store):
    rr = RecordingReranker()
    r = retriever(store, reranker=rr, expander=Expander(store, radius=1), top_n=3)
    result = r.retrieve("PTO carryover days", ENG)

    assert result.dense and result.lexical  # both retrievers ran
    assert all(h.retriever == "hybrid" for h in result.fused)
    assert rr.seen == [len(result.fused)]  # the reranker saw the fused shortlist
    top = result.hits[0]
    assert top.chunk_id == "pto-policy#1"
    assert top.retriever == "rerank"
    # expanded with both neighbours of the carryover chunk
    assert top.neighbor_ids == ["pto-policy#0", "pto-policy#2"]
    assert "overview" in top.text and "HR portal" in top.text
    assert r.search("PTO carryover days", ENG) == result.hits


@pytest.mark.parametrize(
    ("principals", "forbidden"),
    [(ENG, {"comp-bands"}), (HR, {"deploy-runbook"}), (["public"], set(DOCS))],
)
def test_no_stage_ever_returns_a_chunk_the_principal_cannot_see(store, principals, forbidden):
    r = retriever(store, expander=Expander(store, radius=2), top_n=10)
    for q in ["PTO carryover", "compensation bands salary", "deploy rollback Friday", "Atlas"]:
        result = r.retrieve(q, principals)
        stages = [result.dense, result.lexical, result.fused, result.reranked, result.hits]
        docs = {h.doc_id for stage in stages for h in stage}
        neighbor_docs = {n.split("#")[0] for h in result.hits for n in h.neighbor_ids}
        assert not (docs | neighbor_docs) & forbidden, (q, principals)


def test_hr_sees_the_compensation_doc_that_eng_does_not(store):
    r = retriever(store, top_n=5)
    assert "comp-bands" in {h.doc_id for h in r.search("compensation bands", HR)}
    assert "comp-bands" not in {h.doc_id for h in r.search("compensation bands", ENG)}


def test_no_principals_returns_nothing_without_querying(store):
    emb = CountingEmbedder(dim=DIM)
    r = retriever(store, embedder=emb)
    assert r.search("PTO", []) == []
    assert emb.queries == 0


def test_the_reranker_gets_at_most_the_rerank_window(store):
    rr = RecordingReranker()
    r = retriever(store, reranker=rr, rerank_window=2, top_n=1)
    out = r.search("PTO carryover policy deploy", ENG)
    assert rr.seen == [2]
    assert len(out) == 1


def test_window_caps_each_retriever(store):
    r = retriever(store, window=1)
    result = r.retrieve("PTO carryover", ENG)
    assert len(result.dense) <= 1 and len(result.lexical) <= 1


def test_a_zero_weight_skips_that_retriever(store):
    emb = CountingEmbedder(dim=DIM)
    lexical_only = retriever(store, embedder=emb, weights=(0.0, 1.0))
    result = lexical_only.retrieve("PTO carryover", ENG)
    assert emb.queries == 0 and result.dense == [] and result.lexical
    dense_only = retriever(store, weights=(1.0, 0.0)).retrieve("PTO carryover", ENG)
    assert dense_only.lexical == [] and dense_only.dense


def test_filters_reach_both_retrievers(store):
    seen = []

    class Spy(InMemoryStore):
        def dense(self, qvec, principals, filters, k):
            seen.append(("dense", filters))
            return []

        def lexical(self, query, principals, filters, k):
            seen.append(("lexical", filters))
            return []

    r = retriever(Spy())
    r.search("x", ENG, filters={"source": "slack"})
    assert seen == [("dense", {"source": "slack"}), ("lexical", {"source": "slack"})]


def test_top_n_override_per_call(store):
    r = retriever(store, top_n=5)
    assert len(r.search("PTO carryover policy", ENG, top_n=1)) == 1


def test_from_settings_wires_components_by_config_name(store):
    settings = Settings(
        embed={"provider": "fake", "model": "fake-embed", "dim": DIM},
        rerank={"provider": "noop", "top_n": 2, "window": 7},
        retrieval={"window": 11},
        rrf={"k": 30, "weights": [1.0, 2.0]},
        expand={"radius": 0},
    )
    r = HybridRetriever.from_settings(settings, store=store)
    assert isinstance(r.embedder, FakeEmbedder) and isinstance(r.reranker, NoopReranker)
    assert (r.window, r.k, r.weights, r.rerank_window, r.top_n) == (11, 30, (1.0, 2.0), 7, 2)
    assert r.expander is None  # radius 0 turns expansion off
    assert r.search("PTO carryover", ENG)


def test_weights_must_name_both_retrievers(store):
    with pytest.raises(ValueError, match="weights"):
        retriever(store, weights=(1.0,))
