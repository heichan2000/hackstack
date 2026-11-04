"""The retrieval comparison: variants, per-query scoring, the report and the baseline check."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import tests.fake_store  # noqa: F401 - registers store "memory"
from evals.golden import GoldenItem
from evals.run_retrieval import (
    check_baseline,
    format_table,
    no_answer_stats,
    run,
    variant_settings,
)
from hackstack.config import config_hash, load_settings
from hackstack.models import ACL, Chunk, RawDoc
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    return load_settings(REPO / "hackstack.yaml", profile="local", env_file=None)


def test_variants_switch_retrievers_and_the_reranker(settings):
    dense = variant_settings(settings, "dense")
    lexical = variant_settings(settings, "lexical")
    hybrid = variant_settings(settings, "hybrid")
    rerank = variant_settings(settings, "hybrid_rerank")
    assert dense.rrf.weights == [1.0, 0.0] and dense.rerank.provider == "noop"
    assert lexical.rrf.weights == [0.0, 1.0] and lexical.rerank.provider == "noop"
    assert hybrid.rrf.weights == [1.0, 1.0] and hybrid.rerank.provider == "noop"
    assert rerank.rerank.provider == "cross_encoder"  # the profile's own reranker
    hashes = {config_hash(s) for s in (dense, lexical, hybrid, rerank)}
    assert len(hashes) == 4
    assert settings.rrf.weights == [1.0, 1.0]  # the original is untouched
    with pytest.raises(ValueError):
        variant_settings(settings, "nope")


# --- a tiny index -----------------------------------------------------------------------------

EVERYONE = ["group:everyone"]
HR = ["group:hr"]


def _doc(doc_id, texts, acl):
    raw = RawDoc(doc_id=doc_id, source="t", title=doc_id, content="", acl=ACL(principals=acl),
                 updated_at=datetime(2026, 11, 1, tzinfo=UTC))  # fmt: skip
    chunks = [Chunk.for_doc(raw, ordinal=i, text=t) for i, t in enumerate(texts)]
    return raw, chunks


@pytest.fixture
def index():
    store, emb = InMemoryStore(), FakeEmbedder()
    for raw, chunks in [
        _doc("pto", ["carryover five days expire march", "booking time off two weeks"], EVERYONE),
        _doc("deploys", ["two approvals no friday deploys"], EVERYONE),
        _doc("bands", ["senior engineer salary band 128000"], HR),
    ]:
        for c, v in zip(chunks, emb.embed_documents([c.embed_text for c in chunks]), strict=True):
            c.embedding = v
        store.upsert(raw, chunks)
    return store, emb


PRINCIPALS = {"eng": ["user:eng", "group:everyone"], "hr": ["user:hr", "group:everyone",
                                                              "group:hr"]}  # fmt: skip

ITEMS = [
    GoldenItem(id="s1", question="carryover days expire", as_user="eng", kind="synthetic",
               relevant=["pto#0"], relevant_docs=["pto"]),
    GoldenItem(id="h1", question="friday deploys approvals", as_user="eng", kind="handwritten",
               relevant=["deploys#0"], relevant_docs=["deploys"]),
    GoldenItem(id="n1", question="wifi password lisbon", as_user="eng", kind="unanswerable"),
    GoldenItem(id="f1", question="senior engineer salary band", as_user="eng", kind="forbidden",
               forbidden_docs=["bands"]),
]  # fmt: skip


def _run(settings, index, variants=("dense", "lexical", "hybrid")):
    store, emb = index
    return run(settings, ITEMS, variants=variants, store=store, embedder=emb,
               principals_of=PRINCIPALS.__getitem__, golden_sha="abc")  # fmt: skip


def test_run_scores_answerable_items_and_reports_each_variant(settings, index):
    report = _run(settings, index)
    assert report["golden_sha256"] == "abc"
    assert set(report["variants"]) == {"dense", "lexical", "hybrid"}
    for name, v in report["variants"].items():
        assert v["config_hash"] == config_hash(variant_settings(settings, name))
        assert v["metrics"]["n"] == 2  # unanswerable and forbidden are scored separately
        assert v["by_kind"]["synthetic"]["n"] == 1
        assert v["by_kind"]["handwritten"]["n"] == 1
        assert v["forbidden"] == {"n": 1, "leaks": 0}
        assert v["no_answer"]["n"] == 2
        assert len(v["queries"]) == 4
    lex = report["variants"]["lexical"]
    assert lex["metrics"]["recall@1"] == 1.0  # exact word overlap: the lexical store nails these
    assert lex["metrics"]["mrr"] == 1.0
    assert lex["doc_hit@5"] == 1.0


def test_a_leaky_store_is_caught(settings, index):
    store, emb = index

    class Leaky(InMemoryStore):
        def lexical(self, query, principals, filters, k):
            return super().lexical(query, [*principals, "group:hr"], filters, k)

    leaky = Leaky()
    leaky.docs, leaky.chunks = store.docs, store.chunks
    report = run(settings, ITEMS, variants=("lexical",), store=leaky, embedder=emb,
                 principals_of=PRINCIPALS.__getitem__, golden_sha="x")  # fmt: skip
    assert report["variants"]["lexical"]["forbidden"]["leaks"] == 1


def test_no_answer_stats_find_the_separating_threshold():
    answerable = [0.9, 0.8, 0.7, 0.2]
    no_answer = [0.3, 0.25, 0.1]
    s = no_answer_stats(answerable, no_answer)
    assert s["n"] == 3
    assert s["answerable_above_max_no_answer"] == pytest.approx(3 / 4)
    # "answer if top score >= t". t=0.7: 3 answerable kept, 0.2 lost, all 3 no-answer
    # rejected -> 6/7. (t=0.2 keeps all 4 but answers 2 no-answer questions -> 5/7.)
    assert s["best_threshold"] == pytest.approx(0.7)
    assert s["best_threshold_accuracy"] == pytest.approx(6 / 7)


def test_no_answer_stats_with_nothing_to_compare():
    assert no_answer_stats([0.5], [])["n"] == 0


def test_table_lists_every_variant(settings, index):
    table = format_table(_run(settings, index))
    assert "| dense |" in table and "| lexical |" in table and "| hybrid |" in table
    assert "recall@5" in table


def _baseline(**metrics):
    return {"golden_sha256": "abc", "tolerance": 0.0,
            "profiles": {"local": {"lexical": metrics}}}  # fmt: skip


def test_baseline_check_passes_at_or_above(settings, index):
    report = _run(settings, index)
    assert check_baseline(report, _baseline(**{"recall@5": 1.0, "mrr": 1.0}), "local") == []


def test_baseline_check_fails_below(settings, index):
    report = _run(settings, index)
    report["variants"]["lexical"]["metrics"]["recall@5"] = 0.5
    failures = check_baseline(report, _baseline(**{"recall@5": 1.0, "mrr": 1.0}), "local")
    assert failures and "lexical" in failures[0] and "recall@5" in failures[0]


def test_baseline_check_refuses_a_different_golden_set(settings, index):
    report = _run(settings, index)
    report["golden_sha256"] = "other"
    failures = check_baseline(report, _baseline(**{"recall@5": 0.0}), "local")
    assert failures and "golden" in failures[0]


def test_baseline_check_without_an_entry_for_the_profile(settings, index):
    with pytest.raises(KeyError):
        check_baseline(_run(settings, index), _baseline(**{"recall@5": 0.0}), "default")
