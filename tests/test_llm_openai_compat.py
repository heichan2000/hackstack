"""OpenAI-compatible adapter, replaying SSE recorded from the local Ollama (hackstack-chat)."""

import json

import pytest

from hackstack.llm.base import LLMError
from hackstack.llm.openai_compat_provider import OpenAICompatProvider
from hackstack.models import CitationEvent, Stop, TextDelta, ToolCall
from tests.sse_replay import Replay, collect

CITED_ANSWER = (
    "Northwind Labs allows up to **5 unused PTO days** to carry over into the next calendar "
    "year [1]. However, any unused PTO beyond this limit is not paid out [2], and carried-over "
    "days expire on **March 31**."
)


def provider(replay: Replay, **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        model="hackstack-chat",
        base_url="http://ollama.test/v1",
        http_client=replay.client(),
        **kwargs,
    )


def test_streams_text_deltas_then_one_stop_with_usage():
    replay = Replay("openai_cited_answer.sse")

    events = collect(
        provider(replay).stream("Be brief.", [{"role": "user", "content": "PTO?"}], tools=[])
    )

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == CITED_ANSWER
    assert sum(isinstance(e, TextDelta) for e in events) > 10  # streamed, not one blob
    stops = [e for e in events if isinstance(e, Stop)]
    assert stops == [events[-1]]
    assert stops[0].reason == "end_turn"
    assert stops[0].usage == {"input_tokens": 169, "output_tokens": 53}


TOOLS = [
    {
        "name": "search_kb",
        "description": "Search the company knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "cache_control": {"type": "ephemeral"},  # Anthropic-only; must not leak to OpenAI
    },
    {
        "name": "find_experts",
        "description": "Find the people who know most about a topic.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]


def test_parallel_tool_calls_become_tool_call_events_and_an_anthropic_shaped_turn():
    replay = Replay("openai_parallel_tool_calls.sse")

    events = collect(
        provider(replay).stream("sys", [{"role": "user", "content": "PTO and expert?"}], TOOLS)
    )

    calls = [e for e in events if isinstance(e, ToolCall)]
    assert [(c.id, c.name, c.input) for c in calls] == [
        ("call_0qe023ig", "search_kb", {"query": "PTO carryover policy"}),
        ("call_h5ycv3yi", "find_experts", {"topic": "PTO carryover policy"}),
    ]
    stop = events[-1]
    assert isinstance(stop, Stop) and stop.reason == "tool_use"
    # The turn to append to history, unchanged, in the Messages shape the kit uses everywhere.
    assert stop.content == [
        {
            "type": "tool_use",
            "id": "call_0qe023ig",
            "name": "search_kb",
            "input": {"query": "PTO carryover policy"},
        },
        {
            "type": "tool_use",
            "id": "call_h5ycv3yi",
            "name": "find_experts",
            "input": {"topic": "PTO carryover policy"},
        },
    ]


def test_tools_are_sent_as_openai_functions():
    replay = Replay("openai_parallel_tool_calls.sse")

    collect(provider(replay).stream("sys", [{"role": "user", "content": "hi"}], TOOLS))

    assert replay.body["tools"][0] == {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the company knowledge base.",
            "parameters": TOOLS[0]["input_schema"],
        },
    }
    assert replay.body["messages"][0] == {"role": "system", "content": "sys"}


def search_result(source: str, title: str, text: str) -> dict:
    return {
        "type": "search_result",
        "source": source,
        "title": title,
        "content": [{"type": "text", "text": text}],
        "citations": {"enabled": True},
    }


PTO = "Up to 5 unused PTO days carry over into the next calendar year."
PAYOUT = "Unused PTO beyond the carryover limit is not paid out."

HISTORY = [
    {"role": "user", "content": "What is the PTO carryover policy?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},  # Claude-only: dropped
            {"type": "text", "text": "Searching."},
            {"type": "tool_use", "id": "call_1", "name": "search_kb", "input": {"q": "PTO"}},
            {"type": "tool_use", "id": "call_2", "name": "find_experts", "input": {"t": "PTO"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [
                    search_result(
                        "kb://local_files:handbook/pto.md#2", "PTO Policy > Carryover", PTO
                    ),
                    search_result("local_files:finance/faq.md#0", "Finance FAQ > Payouts", PAYOUT),
                ],
            },
            {
                "type": "tool_result",
                "tool_use_id": "call_2",
                "content": "experts index unavailable",
                "is_error": True,
            },
            {"type": "text", "text": "Answer briefly."},
        ],
    },
]


def test_history_is_translated_to_openai_messages_with_numbered_search_results():
    replay = Replay("openai_cited_answer.sse")

    collect(provider(replay).stream("sys", HISTORY, TOOLS))

    assert replay.body["messages"][1:] == [
        {"role": "user", "content": "What is the PTO carryover policy?"},
        {
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_kb", "arguments": '{"q": "PTO"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "find_experts", "arguments": '{"t": "PTO"}'},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Search results. Cite each claim with its number in square brackets, "
            "e.g. [1].\n\n"
            f"[1] PTO Policy > Carryover\n{PTO}\n\n"
            f"[2] Finance FAQ > Payouts\n{PAYOUT}",
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "Error: experts index unavailable"},
        {"role": "user", "content": "Answer briefly."},
    ]


def test_numbered_markers_in_the_answer_become_citations_of_those_search_results():
    replay = Replay("openai_cited_answer.sse")

    events = collect(provider(replay).stream("sys", HISTORY, TOOLS))

    cites = [e.citation for e in events if isinstance(e, CitationEvent)]
    assert [(c.doc_id, c.chunk_id, c.title, c.cited_text) for c in cites] == [
        (
            "local_files:handbook/pto.md",
            "local_files:handbook/pto.md#2",
            "PTO Policy > Carryover",
            PTO,
        ),
        (
            "local_files:finance/faq.md",
            "local_files:finance/faq.md#0",
            "Finance FAQ > Payouts",
            PAYOUT,
        ),
    ]
    # Each citation comes right after the delta that closed its marker ("[", "1", "].").
    first = next(i for i, e in enumerate(events) if isinstance(e, CitationEvent))
    text_before = "".join(e.text for e in events[:first] if isinstance(e, TextDelta))
    assert text_before.endswith("calendar year [1].")
    # The assistant turn carries the text with its markers, ready to append to history.
    assert events[-1].content == [{"type": "text", "text": CITED_ANSWER}]


def test_markers_that_point_at_no_search_result_are_ignored():
    replay = Replay("openai_cited_answer.sse")  # cites [1] and [2]
    one_source = [
        HISTORY[0],
        {"role": "assistant", "content": [HISTORY[1]["content"][2]]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [search_result("a#0", "A", "only source")],
                }
            ],
        },
    ]

    events = collect(provider(replay).stream("sys", one_source, TOOLS))

    assert [e.citation.chunk_id for e in events if isinstance(e, CitationEvent)] == ["a#0"]


def sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def finish(reason: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": reason}]}


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("tool_calls", "end_turn"),  # claimed tool calls, but none arrived: nothing to run
    ],
)
def test_finish_reasons_map_to_messages_stop_reasons(finish_reason, stop_reason):
    events = collect(provider(Replay(sse(finish(finish_reason)))).stream("s", HISTORY[:1], []))

    assert events[-1].reason == stop_reason


def test_a_tool_call_with_broken_json_arguments_keeps_the_raw_text():
    broken = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "function": {"name": "search_kb", "arguments": '{"query": "PTO'},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    events = collect(provider(Replay(sse(broken))).stream("s", HISTORY[:1], TOOLS))

    assert events[0] == ToolCall(
        id="c1", name="search_kb", input={"_invalid_arguments": '{"query": "PTO'}
    )
    assert events[-1].reason == "tool_use"


def test_options_reach_the_request_and_unsupported_ones_do_not():
    replay = Replay("openai_parallel_tool_calls.sse")
    llm = provider(replay, api_key="sk-test", reasoning_effort="none")

    collect(
        llm.stream(
            "s",
            HISTORY[:1],
            TOOLS,
            model="other-model",
            max_tokens=512,
            temperature=0,
            seed=7,
            tool_choice={"type": "auto"},
            cache_control={"type": "ephemeral"},
        )
    )

    body = replay.body
    assert body["model"] == "other-model"
    assert body["max_tokens"] == 512
    assert body["reasoning_effort"] == "none"
    assert (body["temperature"], body["seed"]) == (0, 7)
    assert "tool_choice" not in body  # Ollama ignores it; "auto" is the default everywhere
    assert "cache_control" not in body  # Anthropic-only
    assert replay.headers[0]["authorization"] == "Bearer sk-test"


def test_forced_tool_choice_is_refused():
    llm = provider(Replay("openai_parallel_tool_calls.sse"))

    with pytest.raises(ValueError, match="tool_choice"):
        collect(llm.stream("s", HISTORY[:1], TOOLS, tool_choice={"type": "any"}))


def test_http_errors_raise_llm_error_with_the_server_message():
    body = b'{"error":{"message":"model \\"hackstack-chat\\" not found, try pulling it first"}}'
    llm = provider(Replay(body, status=404))

    with pytest.raises(LLMError, match="not found.*make ollama-models"):
        collect(llm.stream("s", HISTORY[:1], []))
