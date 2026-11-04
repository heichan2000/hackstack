"""Contextual retrieval enrichment, offline: the fake LLM stands in for the aux model."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from hackstack.chunking.markdown import MarkdownChunker
from hackstack.config import EnrichSettings
from hackstack.connectors.local_files import LocalFilesConnector
from hackstack.enrich.contextual import PROMPT_VERSION, ContextualEnricher, clean_context
from hackstack.ingest import Ingestor, ingestor_from_settings
from hackstack.llm.base import LLMError
from hackstack.models import ACL, ParsedDoc, RawDoc, Stop, TextDelta
from hackstack.parsing.router import ParserRouter
from hackstack.sync.manager import SyncManager
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder, FakeLLM
from tests.test_sync import set_mtime, write_corpus

DOC = (
    "# Handbook\n\nNorthwind Labs' employee handbook.\n\n"
    "## PTO\n\nTwenty days a year, carried over until March.\n\n"
    "## Expenses\n\nKeep every receipt and file it within thirty days.\n"
)


def raw_doc(markdown: str = DOC, title: str = "Handbook") -> RawDoc:
    return RawDoc(
        doc_id="local_files:handbook.md",
        source="local_files",
        title=title,
        content=markdown,
        acl=ACL(principals=["group:everyone"]),
    )


def chunks_for(markdown: str = DOC, **chunker: Any):
    raw = raw_doc(markdown)
    parsed = ParsedDoc(markdown=markdown)
    opts = {"size": 20, "max_size": 40, "min_size": 1, **chunker}
    return raw, parsed, MarkdownChunker(**opts).chunk(parsed, raw)


def turn(text: str, **usage: int) -> list:
    return [TextDelta(text=text), Stop(reason="end_turn", usage=usage)]


class ScriptedLLM(FakeLLM):
    """Answers every call with ``reply(call_index)``; records calls like FakeLLM."""

    def __init__(self, reply=lambda i: f"Context {i}.", usage=None) -> None:
        super().__init__(script=[])
        self.reply = reply
        self.usage = usage or {}

    def _next_turn(self, messages):
        i = len(self.calls) - 1
        return turn(self.reply(i), **self.usage)


def user_blocks(call: dict) -> list[dict]:
    (message,) = call["messages"]
    assert message["role"] == "user"
    return message["content"]


# --- the per-chunk call ---------------------------------------------------------------------


def test_each_chunk_gets_a_context_prepended_after_its_breadcrumb():
    raw, parsed, chunks = chunks_for()
    llm = ScriptedLLM()
    texts = [c.text for c in chunks]
    breadcrumbs = [c.embed_text.removesuffix(c.text) for c in chunks]
    assert all(b.startswith("Handbook") and b.endswith("\n\n") for b in breadcrumbs)
    stats = ContextualEnricher(llm).enrich(raw, parsed, chunks)

    assert stats.chunks == len(chunks) == stats.calls > 1
    for i, chunk in enumerate(chunks):
        assert chunk.text == texts[i], "the chunk text itself is never changed"
        assert chunk.context == f"Context {i}."
        assert chunk.embed_text == f"{breadcrumbs[i]}Context {i}.\n\n{chunk.text}"


def test_the_whole_document_is_a_cached_block_before_the_chunk():
    raw, parsed, chunks = chunks_for()
    llm = ScriptedLLM()
    ContextualEnricher(llm).enrich(raw, parsed, chunks)

    systems = {call["system"] for call in llm.calls}
    doc_blocks = {user_blocks(call)[0]["text"] for call in llm.calls}
    assert len(systems) == 1 and len(doc_blocks) == 1, "the shared prefix is byte-identical"
    for call, chunk in zip(llm.calls, chunks, strict=True):
        doc, ask = user_blocks(call)
        assert doc["cache_control"] == {"type": "ephemeral"}
        assert "Twenty days a year" in doc["text"] and "thirty days" in doc["text"]
        assert "cache_control" not in ask
        assert f"<chunk>\n{chunk.text}\n</chunk>" in ask["text"]
        assert call["tools"] == []


def test_model_and_output_limit_are_passed_per_call():
    raw, parsed, chunks = chunks_for()
    llm = ScriptedLLM()
    ContextualEnricher(llm, model="tiny-model", max_output_tokens=300).enrich(raw, parsed, chunks)
    assert all(c["opts"]["model"] == "tiny-model" for c in llm.calls)
    assert all(c["opts"]["max_tokens"] == 300 for c in llm.calls)


def test_a_domain_hint_reaches_the_system_prompt():
    raw, parsed, chunks = chunks_for()
    llm = ScriptedLLM()
    ContextualEnricher(llm, domain_hint="Northwind Labs is a 40-person robotics startup.").enrich(
        raw, parsed, chunks
    )
    assert "40-person robotics startup" in llm.calls[0]["system"]


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  This chunk covers PTO.  ", "This chunk covers PTO."),
        ("<think>\nhmm\n</think>\n\nThis chunk covers PTO.", "This chunk covers PTO."),
        ('"This chunk covers PTO."', "This chunk covers PTO."),
        ("Context: This chunk\n  covers PTO.", "This chunk covers PTO."),
        ("   ", ""),
    ],
)
def test_clean_context(raw, clean):
    assert clean_context(raw, max_tokens=150) == clean


def test_a_rambling_answer_is_capped():
    ramble = " ".join(f"word{i}" for i in range(500))
    assert len(clean_context(ramble, max_tokens=40).split()) <= 40


def test_an_empty_or_refused_answer_leaves_the_chunk_unenriched():
    raw, parsed, chunks = chunks_for()
    before = [c.embed_text for c in chunks]

    class Refuses(FakeLLM):
        def _next_turn(self, messages):
            if len(self.calls) == 1:
                return [Stop(reason="refusal")]
            return turn("   ")

    stats = ContextualEnricher(Refuses(script=[])).enrich(raw, parsed, chunks)
    assert stats.enriched == 0 and stats.failed == len(chunks)
    assert [c.embed_text for c in chunks] == before
    assert all(c.context is None for c in chunks)


def test_a_dead_llm_raises_rather_than_silently_skipping():
    raw, parsed, chunks = chunks_for()

    class Dead(FakeLLM):
        def _next_turn(self, messages):
            raise LLMError("connection refused")

    with pytest.raises(LLMError):
        ContextualEnricher(Dead(script=[])).enrich(raw, parsed, chunks)


# --- documents over the budget ---------------------------------------------------------------


def test_an_over_budget_document_is_split_into_its_top_level_sections():
    long = (
        "# Alpha\n\n" + "alpha " * 60 + "\n\n## Alpha detail\n\nAlpha detail text.\n\n"
        "# Beta\n\n" + "beta " * 60 + "\n"
    )
    raw, parsed, chunks = chunks_for(long, size=80, max_size=120)
    llm = ScriptedLLM()
    stats = ContextualEnricher(llm, max_doc_tokens=100).enrich(raw, parsed, chunks)

    for call, chunk in zip(llm.calls, chunks, strict=True):
        doc = user_blocks(call)[0]["text"]
        top = chunk.section_path[0]
        assert f"# {top}" in doc
        other = "Beta" if top == "Alpha" else "Alpha"
        assert f"# {other}" not in doc, "only the chunk's own part of the document is sent"
    assert stats.split_docs == 1 and stats.skipped == 0


def test_a_chunk_whose_own_section_is_over_budget_is_skipped():
    long = "# Alpha\n\n" + "alpha " * 300 + "\n"
    raw, parsed, chunks = chunks_for(long, size=80, max_size=120)
    llm = ScriptedLLM()
    stats = ContextualEnricher(llm, max_doc_tokens=50).enrich(raw, parsed, chunks)
    assert llm.calls == []
    assert stats.skipped == len(chunks) and stats.enriched == 0
    assert all(c.context is None for c in chunks)


# --- usage and the silent-cache-miss check ---------------------------------------------------


def test_usage_is_summed_across_calls():
    raw, parsed, chunks = chunks_for()
    usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100}
    stats = ContextualEnricher(ScriptedLLM(usage=usage)).enrich(raw, parsed, chunks)
    n = len(chunks)
    assert stats.usage == {
        "input_tokens": 10 * n,
        "output_tokens": 5 * n,
        "cache_read_input_tokens": 100 * n,
    }


def test_a_large_document_that_never_reads_the_cache_logs_a_warning(caplog):
    big = "# Big\n\n" + "\n\n".join(f"## Part {i}\n\n" + "lorem " * 60 for i in range(12))
    raw, parsed, chunks = chunks_for(big, size=80, max_size=120)
    usage = {"input_tokens": 900, "output_tokens": 50, "cache_read_input_tokens": 0}
    with caplog.at_level(logging.WARNING, logger="hackstack.enrich.contextual"):
        ContextualEnricher(ScriptedLLM(usage=usage)).enrich(raw, parsed, chunks)
    assert "cache_read_input_tokens" in caplog.text


def test_no_cache_warning_when_the_provider_does_not_report_cache_usage(caplog):
    big = "# Big\n\n" + "\n\n".join(f"## Part {i}\n\n" + "lorem " * 60 for i in range(12))
    raw, parsed, chunks = chunks_for(big, size=80, max_size=120)
    usage = {"input_tokens": 900, "output_tokens": 50}  # OpenAI-compatible: no cache fields
    with caplog.at_level(logging.WARNING, logger="hackstack.enrich.contextual"):
        ContextualEnricher(ScriptedLLM(usage=usage)).enrich(raw, parsed, chunks)
    assert "cache_read_input_tokens" not in caplog.text


# --- concurrency -----------------------------------------------------------------------------


def test_concurrent_calls_still_put_each_context_on_its_own_chunk():
    raw, parsed, chunks = chunks_for()

    class Echo(FakeLLM):
        def _next_turn(self, messages):
            ask = messages[0]["content"][1]["text"]
            body = ask.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
            return turn(f"About: {body[:12]}")

    ContextualEnricher(Echo(script=[]), concurrency=4).enrich(raw, parsed, chunks)
    for chunk in chunks:
        assert chunk.context == f"About: {chunk.text[:12]}".strip()


# --- settings and the fingerprint the sync manager hashes -------------------------------------


def test_fingerprint_names_the_model_prompt_version_and_budget():
    a = ContextualEnricher(ScriptedLLM(), model="m1", max_doc_tokens=8000)
    assert PROMPT_VERSION in a.fingerprint and "m1" in a.fingerprint
    assert a.fingerprint != ContextualEnricher(ScriptedLLM(), model="m2").fingerprint
    smaller = ContextualEnricher(ScriptedLLM(), model="m1", max_doc_tokens=9)
    assert a.fingerprint != smaller.fingerprint
    hinted = ContextualEnricher(ScriptedLLM(), model="m1", max_doc_tokens=8000, domain_hint="x")
    assert hinted.fingerprint != a.fingerprint


def test_enrichment_is_off_by_default():
    assert EnrichSettings().contextual is False


# --- wired into ingest -----------------------------------------------------------------------


def corpus_ingestor(root, store, *, enricher=None) -> Ingestor:
    conn = LocalFilesConnector()
    conn.configure({"path": str(root)}, {})
    chunker = MarkdownChunker(size=20, max_size=40, min_size=1)
    return Ingestor(
        conn, ParserRouter(), chunker, embedder=FakeEmbedder(), store=store, enricher=enricher
    )


def test_ingest_stores_the_context_and_reports_enrichment(tmp_path):
    root = write_corpus(tmp_path / "corpus")
    store = InMemoryStore()
    report = corpus_ingestor(root, store, enricher=ContextualEnricher(ScriptedLLM())).run()

    assert report.docs == 3 and report.chunks_enriched == report.chunks > 0
    assert all(c.context and c.context in c.embed_text for c in store.chunks.values())


def test_a_dry_run_never_calls_the_llm(tmp_path, monkeypatch):
    root = write_corpus(tmp_path / "corpus")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hackstack.yaml").write_text(
        "llm: {provider: fake}\nenrich: {contextual: true}\n", encoding="utf-8"
    )
    from hackstack.config import load_settings

    settings = load_settings(tmp_path / "hackstack.yaml")
    ing = ingestor_from_settings(settings, "local_files", {}, path=str(root), dry_run=True)
    assert ing.enricher is None


def test_settings_build_an_enricher_on_the_aux_model(tmp_path):
    (tmp_path / "hackstack.yaml").write_text(
        "llm: {provider: fake, aux_model: helper-model}\n"
        "enrich: {contextual: true, max_doc_tokens: 1234, concurrency: 2}\n",
        encoding="utf-8",
    )
    from hackstack.config import load_settings

    settings = load_settings(tmp_path / "hackstack.yaml")
    enricher = ContextualEnricher.from_settings(settings)
    assert enricher.model == "helper-model"
    assert enricher.max_doc_tokens == 1234 and enricher.concurrency == 2

    settings.enrich.model = "override-model"
    assert ContextualEnricher.from_settings(settings).model == "override-model"


def test_turning_enrichment_on_reprocesses_already_indexed_docs(tmp_path):
    root = write_corpus(tmp_path / "corpus")
    for path in root.rglob("*"):
        set_mtime(path, datetime.now(UTC) - timedelta(days=1))
    store = InMemoryStore()

    plain = SyncManager(corpus_ingestor(root, store)).run()
    assert plain.docs == 3
    again = SyncManager(corpus_ingestor(root, store)).run(full=True)
    assert again.unchanged == 3, "enrichment off: the pipeline fingerprint is unchanged"

    enricher = ContextualEnricher(ScriptedLLM(), model="m1")
    on = SyncManager(corpus_ingestor(root, store, enricher=enricher)).run(full=True)
    assert on.docs == 3 and on.unchanged == 0
    assert on.chunks_embedded == on.chunks, "enriched embed_text means fresh vectors"

    steady = SyncManager(
        corpus_ingestor(root, store, enricher=ContextualEnricher(ScriptedLLM(), model="m1"))
    ).run(full=True)
    assert steady.unchanged == 3 and steady.docs == 0
