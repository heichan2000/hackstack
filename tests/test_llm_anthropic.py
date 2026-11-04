"""Anthropic adapter, replaying SSE through the real ``anthropic`` SDK.

``claude_*`` fixtures are hand-written from the Messages streaming docs (no API key to record
them); ``anthropic_compat_*`` was recorded from Ollama's Anthropic-compatible ``/v1/messages``.
"""

import pytest

from hackstack.llm.anthropic_provider import AnthropicProvider
from hackstack.llm.base import LLMError
from hackstack.models import CitationEvent, Stop, TextDelta, ToolCall
from tests.sse_replay import Replay, collect

TOOLS = [
    {
        "name": "search_kb",
        "description": "Search the company knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
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
ASK = [{"role": "user", "content": "What is the PTO carryover policy?"}]


def provider(replay: Replay, **kwargs) -> AnthropicProvider:
    kwargs.setdefault("model", "claude-sonnet-5-5")
    return AnthropicProvider(api_key="test-key", http_client=replay.client(), **kwargs)


def test_streams_text_and_native_search_result_citations():
    events = collect(provider(Replay("claude_cited_answer.sse")).stream("sys", ASK, TOOLS))

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == (
        "Up to 5 unused PTO days carry over into the next year. "
        "Anything beyond that is not paid out."
    )
    cites = [e.citation for e in events if isinstance(e, CitationEvent)]
    assert [(c.doc_id, c.chunk_id, c.title, c.cited_text) for c in cites] == [
        (
            "local_files:handbook/pto.md",
            "local_files:handbook/pto.md#2",
            "PTO Policy > Carryover",
            "Up to 5 unused PTO days carry over into the next calendar year.",
        ),
        (
            "local_files:finance/faq.md",
            "local_files:finance/faq.md#0",
            "Finance FAQ > Payouts",
            "Unused PTO beyond the carryover limit is not paid out.",
        ),
    ]


def test_stop_carries_reason_usage_and_the_turn_unchanged_including_thinking():
    events = collect(provider(Replay("claude_cited_answer.sse")).stream("sys", ASK, TOOLS))

    stop = events[-1]
    assert isinstance(stop, Stop) and sum(isinstance(e, Stop) for e in events) == 1
    assert stop.reason == "end_turn"
    assert stop.usage == {
        "input_tokens": 312,
        "output_tokens": 48,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1840,
    }
    # Thinking blocks must go back to the API exactly as received (signature included).
    assert stop.content[0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "EqQBCkYIBxgCKkBfixture",
    }
    assert stop.content[1]["type"] == "text"
    assert stop.content[1]["citations"][0]["source"] == "kb://local_files:handbook/pto.md#2"


def test_request_uses_top_level_cache_control_and_refusal_fallbacks():
    replay = Replay("claude_cited_answer.sse")

    collect(provider(replay).stream("sys", ASK, TOOLS, max_tokens=2048))

    body = replay.body
    assert body["model"] == "claude-sonnet-5-5"
    assert body["system"] == "sys"
    assert body["messages"] == ASK
    assert body["tools"] == TOOLS
    assert body["max_tokens"] == 2048
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in replay.headers[0]["anthropic-beta"]
    assert "tool_choice" not in body  # auto, the API default


def test_forced_tool_choice_is_refused_before_any_request():
    replay = Replay("claude_cited_answer.sse")

    with pytest.raises(ValueError, match="tool_choice"):
        collect(provider(replay).stream("s", ASK, TOOLS, tool_choice={"type": "any"}))
    assert replay.requests == []


def test_parallel_tool_use_recorded_from_ollama_messages_endpoint():
    llm = provider(
        Replay("anthropic_compat_parallel_tool_use.sse"),
        model="hackstack-chat",
        base_url="http://ollama.test",
    )

    events = collect(llm.stream("sys", ASK, TOOLS))

    calls = [e for e in events if isinstance(e, ToolCall)]
    assert [(c.id, c.name, c.input) for c in calls] == [
        ("call_f8sbdbpw", "search_kb", {"query": "PTO carryover policy"}),
        ("call_4hw79sl8", "find_experts", {"topic": "PTO carryover policy"}),
    ]
    assert events[-1].reason == "tool_use"
    assert [b["type"] for b in events[-1].content] == ["thinking", "tool_use", "tool_use"]


def test_a_custom_base_url_gets_no_fallbacks():
    replay = Replay("anthropic_compat_parallel_tool_use.sse")

    collect(
        provider(replay, model="hackstack-chat", base_url="http://ollama.test").stream(
            "s", ASK, TOOLS
        )
    )

    assert "fallbacks" not in replay.body
    assert "anthropic-beta" not in replay.headers[0]


def test_pause_turn_is_reported_for_the_loop_to_resend():
    events = collect(provider(Replay("claude_pause_turn.sse")).stream("s", ASK, TOOLS))

    assert events[-1].reason == "pause_turn"
    assert events[-1].content == [{"type": "text", "text": "Still working on it."}]


def test_refusal_carries_stop_details():
    events = collect(provider(Replay("claude_refusal.sse")).stream("s", ASK, TOOLS))

    stop = events[-1]
    assert stop.reason == "refusal"
    assert stop.details == {
        "type": "refusal",
        "category": "cyber",
        "explanation": "The request was declined by a safety classifier.",
    }


def test_api_errors_raise_llm_error():
    body = b'{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}'
    llm = provider(Replay(body, status=401), max_retries=0)

    with pytest.raises(LLMError, match="invalid x-api-key"):
        collect(llm.stream("s", ASK, TOOLS))
