"""The agent loop offline, driven by the scripted FakeLLM."""

from __future__ import annotations

import asyncio

import pytest

from hackstack.agent.citations import NO_RESULTS, CitationRegistry
from hackstack.agent.loop import (
    AgentDone,
    ToolEnd,
    ToolStart,
    collect,
    stream_agent,
    stream_single_shot,
)
from hackstack.agent.tools import Passages, Tool, ToolContext, load_tools
from hackstack.models import Citation, CitationEvent, SearchHit, Stop, TextDelta, ToolCall
from tests.fakes import FakeLLM

PRINCIPALS = ["user:eng@northwind.test", "group:everyone", "public"]


def hit(chunk_id: str, text: str = "Five days carry over.", title: str = "PTO Policy"):
    return SearchHit(
        chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], text=text, score=1.0, title=title
    )


class SpyRetriever:
    """Records every search; answers from a query -> hits table."""

    def __init__(self, table: dict[str, list[SearchHit]] | None = None) -> None:
        self.table = table or {}
        self.calls: list[tuple] = []

    def search(self, query, principals, filters=None, *, top_n=None):
        self.calls.append((query, list(principals), filters, top_n))
        return list(self.table.get(query, []))


def ctx(retriever=None) -> ToolContext:
    return ToolContext(principals=PRINCIPALS, retriever=retriever or SpyRetriever())


def run(llm, *, tools=None, context=None, max_iterations=6, question="q?"):
    tools = tools if tools is not None else load_tools(["search_kb", "get_document"])
    return asyncio.run(
        collect(
            stream_agent(
                llm,
                system="sys",
                messages=[{"role": "user", "content": question}],
                tools=tools,
                ctx=context or ctx(),
                max_iterations=max_iterations,
            )
        )
    )


def events_of(llm, **kw):
    async def go():
        return [
            e
            async for e in stream_agent(
                llm,
                system="sys",
                messages=[{"role": "user", "content": "q?"}],
                tools=kw.get("tools") or load_tools(["search_kb"]),
                ctx=kw.get("context") or ctx(),
            )
        ]

    return asyncio.run(go())


def cite(chunk_id: str) -> CitationEvent:
    return CitationEvent(citation=Citation(doc_id=chunk_id.split("#")[0], chunk_id=chunk_id))


# --- parallel tool calls -------------------------------------------------------------------------


def test_parallel_tool_calls_run_concurrently_and_return_in_one_user_message():
    started: list[str] = []
    both = asyncio.Event()

    async def slow(args, context):
        started.append(args["n"])
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=2)  # deadlocks if run one after the other
        return f"result {args['n']}"

    schema = {"type": "object", "properties": {"n": {"type": "string"}}}
    tools = [Tool("slow", "Slow tool.", schema, slow)]
    llm = FakeLLM(
        [
            [
                ToolCall(id="a", name="slow", input={"n": "1"}),
                ToolCall(id="b", name="slow", input={"n": "2"}),
            ],
            [TextDelta(text="done")],
        ]
    )

    result = run(llm, tools=tools)

    assert result.answer == "done" and result.iterations == 2
    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert [b["type"] for b in second[1]["content"]] == ["tool_use", "tool_use"]
    results = second[2]["content"]
    assert [(b["type"], b["tool_use_id"], b["content"]) for b in results] == [
        ("tool_result", "a", "result 1"),
        ("tool_result", "b", "result 2"),
    ]
    assert [t.name for t in result.tool_calls] == ["slow", "slow"]


def test_search_results_become_numbered_search_result_blocks_and_citations_resolve():
    retriever = SpyRetriever(
        {
            "pto": [hit("pto#1"), hit("pto#2", "Requests go through HR.")],
            "deploy": [hit("deploy#0", "No Friday deploys.", "Deploy Runbook")],
        }
    )
    llm = FakeLLM(
        [
            [
                ToolCall(id="a", name="search_kb", input={"query": "pto"}),
                ToolCall(
                    id="b", name="search_kb", input={"query": "deploy", "since": "2026-11-01"}
                ),
            ],
            [
                TextDelta(text="Five days [1]; no Friday deploys [3]."),
                cite("pto#1"),
                cite("deploy#0"),
            ],
        ]
    )

    result = run(llm, context=ctx(retriever))

    results = llm.calls[1]["messages"][2]["content"]
    first = results[0]["content"]
    assert [b["type"] for b in first] == ["search_result", "search_result"]
    assert first[0]["source"] == "kb://pto#1" and first[0]["title"] == "PTO Policy"
    assert retriever.calls[1][2] == {"source": None, "since": "2026-11-01"}
    assert [s.number for s in result.sources] == [1, 2, 3]
    assert [c.chunk_id for c in result.citations] == ["pto#1", "deploy#0"]
    assert result.citations[1].title == "Deploy Runbook"
    assert [s.cited for s in result.sources] == [True, False, True]
    assert not result.low_grounding


def test_empty_search_returns_plain_no_results_text():
    llm = FakeLLM(
        [
            [ToolCall(id="a", name="search_kb", input={"query": "nothing"})],
            [TextDelta(text="I don't know.")],
        ]
    )
    result = run(llm)
    tool_result = llm.calls[1]["messages"][2]["content"][0]
    assert tool_result["content"] == [{"type": "text", "text": NO_RESULTS}]
    assert "is_error" not in tool_result
    assert result.low_grounding


def test_passages_already_shown_are_not_sent_again():
    retriever = SpyRetriever({"pto": [hit("pto#1")], "pto again": [hit("pto#1")]})
    llm = FakeLLM(
        [
            [ToolCall(id="a", name="search_kb", input={"query": "pto"})],
            [ToolCall(id="b", name="search_kb", input={"query": "pto again"})],
            [TextDelta(text="ok")],
        ]
    )
    result = run(llm, context=ctx(retriever))
    again = llm.calls[2]["messages"][4]["content"][0]["content"]
    assert again == [
        {"type": "text", "text": "No new results: every match was already shown above."}
    ]
    assert len(result.sources) == 1


# --- iteration cap -----------------------------------------------------------------------------


def test_iteration_cap_forces_an_answer_on_the_last_call():
    search = ToolCall(id="x", name="search_kb", input={"query": "q"})
    llm = FakeLLM(
        [
            [search.model_copy(update={"id": "1"})],
            [search.model_copy(update={"id": "2"})],
            [TextDelta(text="best effort")],
        ]
    )
    result = run(llm, max_iterations=3)
    assert len(llm.calls) == 3
    assert "tool_choice" not in llm.calls[0]["opts"]
    assert llm.calls[2]["opts"]["tool_choice"] == {"type": "none"}
    assert result.answer == "best effort" and result.stop_reason == "end_turn"


def test_iteration_cap_stops_when_the_model_keeps_calling_tools():
    calls = [[ToolCall(id=str(i), name="search_kb", input={"query": "q"})] for i in range(3)]
    retriever = SpyRetriever()
    llm = FakeLLM(calls)
    result = run(llm, max_iterations=3, context=ctx(retriever))
    assert len(llm.calls) == 3 and len(retriever.calls) == 2  # the last calls were not run
    assert result.stop_reason == "max_iterations"
    last = result.messages[-1]
    assert last["role"] == "user" and last["content"][0]["is_error"]  # history stays valid


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError):
        run(FakeLLM([]), max_iterations=0)


# --- stop reasons ------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["end_turn", "stop_sequence"])
def test_done_reasons_end_the_loop(reason):
    llm = FakeLLM([[TextDelta(text="hi"), Stop(reason=reason)]])
    result = run(llm)
    assert result.stop_reason == reason and result.iterations == 1
    assert result.messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}


def test_pause_turn_appends_the_turn_verbatim_and_resends():
    paused = [{"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}}]
    llm = FakeLLM(
        [
            [Stop(reason="pause_turn", content=paused)],
            [TextDelta(text="resumed"), Stop(reason="end_turn")],
        ]
    )
    result = run(llm)
    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1] == {"role": "assistant", "content": paused}
    assert result.answer == "resumed"


def test_stop_content_is_appended_unchanged_not_rebuilt():
    content = [
        {"type": "thinking", "thinking": "...", "signature": "sig"},
        {"type": "tool_use", "id": "t", "name": "search_kb", "input": {"query": "q"}},
    ]
    llm = FakeLLM(
        [
            [
                ToolCall(id="t", name="search_kb", input={"query": "q"}),
                Stop(reason="tool_use", content=content),
            ],
            [TextDelta(text="ok")],
        ]
    )
    run(llm)
    assert llm.calls[1]["messages"][1] == {"role": "assistant", "content": content}


def test_max_tokens_marks_truncation():
    llm = FakeLLM([[TextDelta(text="partial"), Stop(reason="max_tokens")]])
    result = run(llm)
    assert result.truncated and result.stop_reason == "max_tokens" and result.answer == "partial"


def test_max_tokens_mid_tool_call_does_not_run_the_tool():
    retriever = SpyRetriever()
    llm = FakeLLM(
        [
            [
                TextDelta(text="Let me"),
                ToolCall(id="t", name="search_kb", input={"query": "q"}),
                Stop(reason="max_tokens"),
            ]
        ]
    )
    result = run(llm, context=ctx(retriever))
    assert retriever.calls == []
    assert all(b["type"] != "tool_use" for b in result.messages[-1]["content"])
    assert result.truncated


def test_refusal_surfaces_the_details():
    details = {"category": "cyber", "explanation": "no"}
    llm = FakeLLM([[Stop(reason="refusal", details=details)]])
    result = run(llm)
    assert result.stop_reason == "refusal" and result.refusal == details
    assert result.messages == [{"role": "user", "content": "q?"}]  # empty turn not appended


def test_context_window_exceeded_ends_with_the_reason():
    llm = FakeLLM([[TextDelta(text="..."), Stop(reason="model_context_window_exceeded")]])
    result = run(llm)
    assert result.stop_reason == "model_context_window_exceeded" and len(llm.calls) == 1


# --- tool errors -------------------------------------------------------------------------------


def test_bad_calls_become_error_results_not_crashes():
    def boom(args, context):
        raise RuntimeError("db down")

    tools = [*load_tools(["search_kb"]), Tool("boom", "Breaks.", {"type": "object"}, boom)]
    llm = FakeLLM(
        [
            [
                ToolCall(id="1", name="nope", input={}),
                ToolCall(id="2", name="search_kb", input={"_invalid_arguments": "{query"}),
                ToolCall(id="3", name="search_kb", input={"query": ""}),
                ToolCall(id="4", name="search_kb", input={"query": "x", "since": "last week"}),
                ToolCall(id="5", name="boom", input={}),
            ],
            [TextDelta(text="sorry")],
        ]
    )
    result = run(llm, tools=tools)
    results = llm.calls[1]["messages"][2]["content"]
    assert [r["is_error"] for r in results] == [True] * 5
    assert "Unknown tool 'nope'" in results[0]["content"]
    assert "not valid JSON" in results[1]["content"]
    assert "non-empty `query`" in results[2]["content"]
    assert "`since` must be a date" in results[3]["content"]
    assert "db down" in results[4]["content"]
    assert all(t.is_error for t in result.tool_calls)


# --- low grounding and streaming --------------------------------------------------------------


def test_low_grounding_when_the_answer_cites_nothing():
    result = run(FakeLLM([[TextDelta(text="Probably five days.")]]))
    assert result.low_grounding and result.citations == []


def test_events_stream_text_tool_progress_and_end_with_done():
    retriever = SpyRetriever({"pto": [hit("pto#1")]})
    llm = FakeLLM(
        [
            [ToolCall(id="a", name="search_kb", input={"query": "pto"})],
            [TextDelta(text="Five [1]."), cite("pto#1")],
        ]
    )
    events = events_of(llm, context=ctx(retriever))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolStart", "ToolEnd", "TextDelta", "CitationEvent", "AgentDone"]
    assert isinstance(events[0], ToolStart) and events[0].input == {"query": "pto"}
    assert isinstance(events[1], ToolEnd) and events[1].results == 1
    assert events[3].citation.title == "PTO Policy"  # resolved through the registry
    assert isinstance(events[-1], AgentDone)


def test_numbering_continues_from_the_stored_history():
    reg = CitationRegistry()
    old = [reg.add(hit("old#0"))]
    history = [
        {"role": "user", "content": "first?"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "o", "name": "search_kb", "input": {"query": "x"}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "o", "content": old}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        {"role": "user", "content": "second?"},
    ]
    context = ToolContext(
        principals=PRINCIPALS,
        retriever=SpyRetriever({"pto": [hit("pto#1"), hit("old#0")]}),
        citations=CitationRegistry.from_messages(history),
    )
    llm = FakeLLM(
        [[ToolCall(id="a", name="search_kb", input={"query": "pto"})], [TextDelta(text="ok")]]
    )
    result = asyncio.run(
        collect(
            stream_agent(
                llm, system="s", messages=history, tools=load_tools(["search_kb"]), ctx=context
            )
        )
    )
    assert [(s.number, s.chunk_id) for s in result.sources] == [(1, "old#0"), (2, "pto#1")]


# --- single-shot ---------------------------------------------------------------------------------


def single_shot(llm, retriever):
    return asyncio.run(
        collect(
            stream_single_shot(
                llm, system="sys", messages=[{"role": "user", "content": "pto"}], ctx=ctx(retriever)
            )
        )
    )


def test_single_shot_retrieves_once_and_answers_in_one_call_without_tools():
    retriever = SpyRetriever({"pto": [hit("pto#1")]})
    llm = FakeLLM([[TextDelta(text="Five days [1]."), cite("pto#1")]])
    result = single_shot(llm, retriever)
    assert len(llm.calls) == 1 and llm.calls[0]["tools"] == []
    content = llm.calls[0]["messages"][-1]["content"]
    assert [b["type"] for b in content] == ["search_result", "text"]
    assert content[-1]["text"] == "pto"
    assert len(retriever.calls) == 1 and retriever.calls[0][1] == PRINCIPALS
    assert result.mode == "single_shot" and not result.low_grounding
    assert result.iterations == 1


def test_single_shot_with_no_results_sends_no_results_text():
    llm = FakeLLM([[TextDelta(text="I don't know.")]])
    result = single_shot(llm, SpyRetriever())
    content = llm.calls[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": NO_RESULTS}
    assert result.low_grounding


def test_tool_handlers_may_return_passages_directly():
    def finder(args, context):
        return Passages(hits=[hit("people#0", "Ada knows PTO.", "Team Directory")])

    tools = [Tool("find_experts", "Finds people.", {"type": "object"}, finder)]
    llm = FakeLLM([[ToolCall(id="f", name="find_experts", input={})], [TextDelta(text="Ada")]])
    result = run(llm, tools=tools)
    assert result.sources[0].title == "Team Directory"
