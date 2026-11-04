"""Core tools against the in-memory store: search_kb, get_document, app tool loading."""

from __future__ import annotations

import pytest

from hackstack.agent.citations import CitationRegistry
from hackstack.agent.tools import (
    GET_DOCUMENT,
    SEARCH_KB,
    Passages,
    ToolContext,
    load_tools,
)
from hackstack.models import ACL, Chunk, RawDoc
from hackstack.registry import available, get
from hackstack.retrieval.hybrid import HybridRetriever
from hackstack.retrieval.rerank import NoopReranker
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder

ENG = ["user:eng@northwind.test", "group:everyone", "group:engineering", "public"]
HR = ["user:hr@northwind.test", "group:everyone", "group:hr", "public"]


@pytest.fixture
def retriever():
    store = InMemoryStore()
    emb = FakeEmbedder(dim=64)
    docs = {
        "pto": (
            "PTO Policy",
            ["group:everyone"],
            ["PTO overview.", "Five days carry over.", "Ask HR two weeks ahead."],
        ),
        "comp": ("Compensation Bands", ["group:hr"], ["Bands L3 to L7.", "Reviewed in March."]),
    }
    for doc_id, (title, acl, texts) in docs.items():
        raw = RawDoc(
            doc_id=doc_id, source="local_files", title=title, content="", acl=ACL(principals=acl)
        )
        chunks = [Chunk.for_doc(raw, ordinal=i, text=t) for i, t in enumerate(texts)]
        for c in chunks:
            c.embedding = emb.embed_query(c.text)
        store.upsert(raw, chunks)
    return HybridRetriever(store, emb, NoopReranker(), window=10, top_n=3)


def call(name, args, ctx):
    return get("tool", name).handler(args, ctx)


def test_core_tools_have_multi_sentence_descriptions_and_schemas():
    for tool in load_tools([SEARCH_KB, GET_DOCUMENT]):
        assert tool.description.count(". ") >= 3
        d = tool.definition()
        assert d["input_schema"]["type"] == "object" and d["input_schema"]["required"]


def test_search_kb_is_acl_filtered(retriever):
    eng = call(SEARCH_KB, {"query": "bands"}, ToolContext(principals=ENG, retriever=retriever))
    hr = call(SEARCH_KB, {"query": "bands"}, ToolContext(principals=HR, retriever=retriever))
    assert isinstance(eng, Passages)
    assert "comp" not in {h.doc_id for h in eng.hits}
    assert "comp" in {h.doc_id for h in hr.hits}


def test_get_document_by_title_returns_passages_in_order(retriever):
    reg = CitationRegistry()
    first = retriever.search("carry over", ENG)[0]
    reg.add(first)
    ctx = ToolContext(principals=ENG, retriever=retriever, citations=reg)
    out = call(GET_DOCUMENT, {"document": "PTO Policy"}, ctx)
    assert [h.chunk_id for h in out.hits] == ["pto#0", "pto#1", "pto#2"]
    assert out.skip_seen is False  # a full read may repeat a passage already shown


def test_get_document_by_source_id_and_fails_closed(retriever):
    eng = ToolContext(principals=ENG, retriever=retriever)
    assert call(GET_DOCUMENT, {"document": "kb://pto#2"}, eng).hits[0].doc_id == "pto"
    denied = call(GET_DOCUMENT, {"document": "comp"}, eng)
    assert denied.hits == [] and "can read" in denied.empty
    hr = ToolContext(principals=HR, retriever=retriever)
    assert len(call(GET_DOCUMENT, {"document": "comp"}, hr).hits) == 2


def test_get_document_respects_its_character_budget(retriever):
    ctx = ToolContext(
        principals=ENG, retriever=retriever, document_chars=40
    )  # 13 + 21 fit, 23 more do not
    out = call(GET_DOCUMENT, {"document": "pto"}, ctx)
    assert len(out.hits) == 2 and out.hits[-1].text.endswith("[document truncated]")


def test_app_tools_load_from_the_app_folder_and_unknown_names_are_skipped(tmp_path, caplog):
    (tmp_path / "tools.py").write_text(
        "from hackstack.agent.tools import tool\n"
        "@tool('hello_tool', 'Says hello. It is a test tool. Use it never. Really.',"
        " {'type': 'object'})\n"
        "def hello(args, ctx):\n    return 'hello'\n",
        encoding="utf-8",
    )
    tools = load_tools(["search_kb", "hello_tool", "not_built_yet"], app_root=tmp_path)
    assert [t.name for t in tools] == ["search_kb", "hello_tool"]
    assert "hello_tool" in available("tool")
    assert "not_built_yet" in caplog.text
