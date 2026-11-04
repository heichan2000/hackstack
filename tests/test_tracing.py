"""Tracing offline: an in-memory span exporter asserts the span tree of one request."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

# The SDK and OTLP exporter are the optional ``tracing`` extra (``make setup`` installs it).
pytest.importorskip("opentelemetry.sdk.trace", reason="needs the tracing extra")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from hackstack import semconv as sc
from hackstack import tracing
from hackstack.agent.loop import collect, stream_agent, stream_single_shot
from hackstack.agent.tools import ToolContext, load_tools
from hackstack.config import Settings, config_hash
from hackstack.models import ACL, Chunk, RawDoc, Stop, TextDelta, ToolCall
from hackstack.retrieval.expand import Expander
from hackstack.retrieval.hybrid import HybridRetriever
from hackstack.retrieval.rerank import NoopReranker
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder, FakeLLM

DIM = 64
ENG = ["user:eng@northwind.test", "group:everyone", "public"]
QUESTION = "What is the PTO carryover policy?"


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    tracing.use_provider(provider)
    yield exp
    tracing.reset()


@pytest.fixture
def content_exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    tracing.use_provider(provider, capture_content=True)
    yield exp
    tracing.reset()


@pytest.fixture
def retriever():
    store = InMemoryStore()
    emb = FakeEmbedder(dim=DIM)
    raw = RawDoc(
        doc_id="pto-policy",
        source="local_files",
        title="PTO Policy",
        content="",
        acl=ACL(principals=["group:everyone"]),
    )
    texts = [
        "Northwind PTO policy overview.",
        "PTO carryover: up to five unused days carry over.",
        "Requests go through the HR portal.",
    ]
    chunks = [Chunk.for_doc(raw, ordinal=i, text=t) for i, t in enumerate(texts)]
    for c, v in zip(chunks, emb.embed_documents([c.embed_text for c in chunks]), strict=True):
        c.embedding = v
    store.upsert(raw, chunks)
    return HybridRetriever(
        store=store,
        embedder=emb,
        reranker=NoopReranker(),
        expander=Expander(store, radius=1),
        window=10,
        rerank_window=10,
        top_n=3,
    )


def two_searches_then_answer() -> FakeLLM:
    return FakeLLM(
        [
            [
                ToolCall(id="c1", name="search_kb", input={"query": "PTO carryover"}),
                ToolCall(id="c2", name="search_kb", input={"query": "HR portal"}),
                Stop(reason="tool_use", usage={"input_tokens": 120, "output_tokens": 30}),
            ],
            [
                TextDelta(text="Five days carry over [1]."),
                Stop(reason="end_turn", usage={"input_tokens": 400, "output_tokens": 12}),
            ],
        ]
    )


def run_agent(llm, retriever, **kw):
    ctx = ToolContext(principals=ENG, retriever=retriever)
    return asyncio.run(
        collect(
            stream_agent(
                llm,
                system="sys",
                messages=[{"role": "user", "content": QUESTION}],
                tools=load_tools(["search_kb"]),
                ctx=ctx,
                **kw,
            )
        )
    )


def by_name(spans):
    return {s.name: s for s in spans}


def children(spans, parent):
    return [s for s in spans if s.parent is not None and s.parent.span_id == parent.context.span_id]


# --- the span tree -------------------------------------------------------------------------------


def test_one_agent_turn_is_one_trace_with_the_full_span_tree(exporter, retriever):
    run_agent(two_searches_then_answer(), retriever, trace_attrs={sc.CONVERSATION_ID: "conv-1"})
    spans = exporter.get_finished_spans()

    assert len({s.context.trace_id for s in spans}) == 1  # one trace per request
    roots = [s for s in spans if s.parent is None]
    assert [r.name for r in roots] == ["invoke_agent"]
    root = roots[0]
    assert root.attributes[sc.OPERATION_NAME] == sc.OP_INVOKE_AGENT
    assert root.attributes[sc.OI_SPAN_KIND] == "AGENT"
    assert root.attributes[sc.CONVERSATION_ID] == "conv-1"
    assert root.attributes[sc.HS_MODE] == "agent"

    top = children(spans, root)
    assert [s.name for s in top if s.name.startswith("chat")] == ["chat", "chat"]  # 2 LLM calls
    tools = [s for s in top if s.name == "execute_tool search_kb"]
    assert len(tools) == 2  # the parallel calls, each its own span under the agent
    assert {t.attributes[sc.TOOL_CALL_ID] for t in tools} == {"c1", "c2"}

    for tool in tools:
        (retrieval,) = children(spans, tool)
        assert retrieval.name == "retrieval"
        assert retrieval.attributes[sc.OPERATION_NAME] == sc.OP_RETRIEVAL
        stages = [s.name for s in children(spans, retrieval)]
        assert stages == ["embeddings fake-embed", "dense", "lexical", "rrf", "rerank", "expand"]


def test_llm_spans_carry_model_usage_and_finish_reason(exporter, retriever):
    llm = two_searches_then_answer()
    llm.model = "hackstack-chat"
    llm.provider_name = "openai"
    run_agent(llm, retriever)
    chats = [s for s in exporter.get_finished_spans() if s.name.startswith("chat")]
    first, second = sorted(chats, key=lambda s: s.start_time)
    assert first.name == "chat hackstack-chat"
    assert first.attributes[sc.OPERATION_NAME] == sc.OP_CHAT
    assert first.attributes[sc.PROVIDER_NAME] == "openai"
    assert first.attributes[sc.REQUEST_MODEL] == "hackstack-chat"
    assert first.attributes[sc.USAGE_INPUT_TOKENS] == 120
    assert first.attributes[sc.USAGE_OUTPUT_TOKENS] == 30
    assert first.attributes[sc.FINISH_REASONS] == ("tool_use",)
    assert first.attributes[sc.OI_SPAN_KIND] == "LLM"
    assert second.attributes[sc.FINISH_REASONS] == ("end_turn",)
    assert second.attributes[sc.HS_ITERATION] == 2


def test_retrieval_spans_record_stage_sizes_and_document_ids(exporter, retriever):
    run_agent(two_searches_then_answer(), retriever)
    spans = exporter.get_finished_spans()
    retrieval = next(s for s in spans if s.name == "retrieval")
    docs = json.loads(retrieval.attributes[sc.RETRIEVAL_DOCUMENTS])
    assert docs and all(set(d) == {"id", "score"} for d in docs)
    assert retrieval.attributes[sc.HS_PRINCIPALS] == len(ENG)
    rerank = next(s for s in spans if s.name == "rerank")
    assert rerank.attributes[sc.OI_SPAN_KIND] == "RERANKER"
    assert rerank.attributes[sc.HS_INPUT_COUNT] >= rerank.attributes[sc.HS_OUTPUT_COUNT]
    dense = next(s for s in spans if s.name == "dense")
    assert dense.attributes[sc.HS_WINDOW] == 10
    embed = next(s for s in spans if s.name.startswith("embeddings"))
    assert embed.attributes[sc.OPERATION_NAME] == sc.OP_EMBEDDINGS
    assert embed.attributes[sc.EMBEDDING_DIMENSIONS] == DIM


def test_single_shot_traces_retrieval_then_one_llm_call(exporter, retriever):
    llm = FakeLLM([[TextDelta(text="Five days [1]."), Stop(reason="end_turn")]])
    ctx = ToolContext(principals=ENG, retriever=retriever)
    asyncio.run(
        collect(
            stream_single_shot(
                llm, system="sys", messages=[{"role": "user", "content": QUESTION}], ctx=ctx
            )
        )
    )
    spans = exporter.get_finished_spans()
    (root,) = [s for s in spans if s.parent is None]
    assert root.attributes[sc.HS_MODE] == "single_shot"
    assert sorted(s.name for s in children(spans, root)) == ["chat", "retrieval"]


# --- content capture (off by default: spans would carry PII) -------------------------------------


def _all_attribute_text(spans) -> str:
    return " ".join(str(v) for s in spans for v in s.attributes.values())


def test_no_prompt_query_or_answer_text_on_spans_by_default(exporter, retriever):
    run_agent(two_searches_then_answer(), retriever)
    text = _all_attribute_text(exporter.get_finished_spans())
    for secret in (QUESTION, "PTO carryover", "HR portal", "Five days carry over", "sys"):
        assert secret not in text


def test_capture_content_puts_question_queries_and_answer_on_spans(content_exporter, retriever):
    run_agent(two_searches_then_answer(), retriever)
    spans = content_exporter.get_finished_spans()
    root = next(s for s in spans if s.parent is None)
    assert root.attributes[sc.OI_INPUT_VALUE] == QUESTION
    assert root.attributes[sc.OI_OUTPUT_VALUE] == "Five days carry over [1]."
    tool = next(s for s in spans if s.attributes.get(sc.TOOL_CALL_ID) == "c1")
    assert json.loads(tool.attributes[sc.TOOL_CALL_ARGUMENTS]) == {"query": "PTO carryover"}
    retrieval = children(spans, tool)[0]
    assert retrieval.attributes[sc.RETRIEVAL_QUERY_TEXT] == "PTO carryover"


# --- errors --------------------------------------------------------------------------------------


def test_a_failing_tool_marks_its_span_as_an_error(exporter, retriever):
    class Broken:
        def search(self, *a, **kw):
            raise RuntimeError("store down")

    llm = FakeLLM(
        [
            [ToolCall(id="c1", name="search_kb", input={"query": "x"})],
            [TextDelta(text="Sorry."), Stop(reason="end_turn")],
        ]
    )
    run_agent(llm, Broken())
    tool = next(s for s in exporter.get_finished_spans() if s.name == "execute_tool search_kb")
    assert tool.status.status_code.name == "ERROR"
    assert tool.attributes[sc.ERROR_TYPE] == "tool_error"


def test_a_failing_llm_call_ends_both_spans_with_error(exporter, retriever):
    class Down:
        async def stream(self, *a, **kw):
            raise ConnectionError("ollama is not running")
            yield  # pragma: no cover

    with pytest.raises(ConnectionError):
        run_agent(Down(), retriever)
    spans = by_name(exporter.get_finished_spans())
    assert spans["chat"].status.status_code.name == "ERROR"
    assert spans["chat"].attributes[sc.ERROR_TYPE] == "ConnectionError"
    assert spans["invoke_agent"].status.status_code.name == "ERROR"


def test_a_consumer_that_stops_early_still_ends_the_root_span(exporter, retriever):
    async def first_event_only():
        ctx = ToolContext(principals=ENG, retriever=retriever)
        gen = stream_agent(
            two_searches_then_answer(),
            system="sys",
            messages=[{"role": "user", "content": QUESTION}],
            tools=load_tools(["search_kb"]),
            ctx=ctx,
        )
        async for _ in gen:
            break
        await gen.aclose()  # what Starlette does when the SSE client goes away

    asyncio.run(first_event_only())
    names = [s.name for s in exporter.get_finished_spans()]
    assert "invoke_agent" in names


# --- disabled / setup ----------------------------------------------------------------------------


def test_disabled_tracing_installs_nothing_and_spans_are_no_ops(retriever):
    tracing.reset()
    assert tracing.setup_tracing(Settings()) is None
    assert not tracing.is_enabled()
    result = run_agent(two_searches_then_answer(), retriever)  # no provider: still works
    assert result.answer == "Five days carry over [1]."


def test_enabled_without_the_tracing_extra_warns_and_runs_untraced(monkeypatch, caplog):
    tracing.reset()
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)  # import now fails
    settings = Settings(tracing={"enabled": True})
    with caplog.at_level("WARNING"):
        assert tracing.setup_tracing(settings) is None
    assert "uv sync --extra tracing" in caplog.text
    assert not tracing.is_enabled()


def test_build_provider_tags_the_resource_for_phoenix():
    settings = Settings(tracing={"enabled": True, "project": "demo"})
    exp = InMemorySpanExporter()
    provider = tracing.build_provider(settings, exporter=exp)
    attrs = provider.resource.attributes
    assert attrs["service.name"] == "hackstack"
    assert attrs[sc.OI_PROJECT_NAME] == "demo"
    assert attrs[sc.HS_CONFIG_HASH] == config_hash(settings)
    assert attrs[sc.HS_PROFILE] == settings.profile
    provider.shutdown()


def test_setup_is_idempotent_and_sets_the_global_provider_once(monkeypatch):
    # The SDK can arrive through other packages; the exporter only comes with the extra.
    pytest.importorskip(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter", reason="needs the tracing extra"
    )
    tracing.reset()
    installed = []
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", installed.append)
    settings = Settings(tracing={"enabled": True, "endpoint": "http://127.0.0.1:9/v1/traces"})
    first = tracing.setup_tracing(settings)
    second = tracing.setup_tracing(settings)
    assert first is not None and first is second
    assert installed == [first]
    assert tracing.is_enabled()
    tracing.reset()
    first.shutdown()


def test_the_exporter_sends_the_phoenix_api_key_only_when_set():
    from pydantic import SecretStr

    from hackstack.config import Secrets

    assert tracing.exporter_headers(Secrets(_env_file=None)) == {}
    keyed = Secrets(_env_file=None, phoenix_api_key=SecretStr("px-123"))
    assert tracing.exporter_headers(keyed) == {"authorization": "Bearer px-123"}
