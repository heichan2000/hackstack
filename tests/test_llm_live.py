"""Both adapters against the real local Ollama. Skipped unless it is up with ``hackstack-chat``
(``make ollama-models``). Run just these: ``uv run pytest -m ollama``.
"""

from __future__ import annotations

import os

import httpx2
import pytest

from hackstack.llm.anthropic_provider import AnthropicProvider
from hackstack.llm.openai_compat_provider import OpenAICompatProvider
from hackstack.models import CitationEvent, Stop, TextDelta, ToolCall
from tests.sse_replay import collect

pytestmark = pytest.mark.ollama

MODEL = "hackstack-chat"
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM = (
    "You answer questions about Northwind Labs using the tools. When several tools are needed "
    "and they do not depend on each other, call them all in the same turn. Answer only from "
    "tool results."
)
TOOLS = [
    {
        "name": "search_kb",
        "description": "Search the company knowledge base for policies and documents.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "find_experts",
        "description": "Find the people at Northwind who know most about a topic.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]
QUESTION = (
    "What is the PTO carryover policy, and who is the expert on it? "
    "Use search_kb and find_experts at the same time."
)
DETERMINISTIC = {"temperature": 0, "seed": 7}


@pytest.fixture(scope="module", autouse=True)
def _ollama_with_model():
    try:
        tags = httpx2.get(f"{OLLAMA}/api/tags", timeout=2).json()
    except httpx2.HTTPError as exc:
        pytest.skip(f"Ollama not reachable at {OLLAMA}: {exc}")
    if not any(m["name"].split(":")[0] == MODEL for m in tags.get("models", [])):
        pytest.skip(f"Ollama has no {MODEL} model; run `make ollama-models`")


def openai_compat() -> OpenAICompatProvider:
    return OpenAICompatProvider(model=MODEL, base_url=f"{OLLAMA}/v1", reasoning_effort="none")


def test_openai_compat_streams_text():
    events = collect(
        openai_compat().stream(
            "Reply in one short sentence.",
            [{"role": "user", "content": "Say hello to the Northwind team."}],
            [],
            **DETERMINISTIC,
        )
    )

    assert sum(isinstance(e, TextDelta) for e in events) > 1
    assert isinstance(events[-1], Stop) and events[-1].reason == "end_turn"
    assert events[-1].usage["output_tokens"] > 0


def test_openai_compat_parallel_tool_calls_round_trip_with_citations():
    llm = openai_compat()
    history: list[dict] = [{"role": "user", "content": QUESTION}]

    first = collect(llm.stream(SYSTEM, history, TOOLS, **DETERMINISTIC))

    calls = [e for e in first if isinstance(e, ToolCall)]
    assert sorted(c.name for c in calls) == ["find_experts", "search_kb"]  # both, one turn
    assert first[-1].reason == "tool_use"

    results = {
        "search_kb": [
            {
                "type": "search_result",
                "source": "kb://local_files:handbook/pto.md#2",
                "title": "PTO Policy > Carryover",
                "content": [
                    {
                        "type": "text",
                        "text": "Up to 5 unused PTO days carry over into the next calendar "
                        "year. Carried days expire on March 31.",
                    }
                ],
                "citations": {"enabled": True},
            }
        ],
        "find_experts": "Priya Raman (People Ops) owns the PTO policy.",
    }
    history += [
        {"role": "assistant", "content": first[-1].content},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": c.id, "content": results[c.name]}
                for c in calls
            ],
        },
    ]

    second = collect(llm.stream(SYSTEM, history, TOOLS, **DETERMINISTIC))

    answer = "".join(e.text for e in second if isinstance(e, TextDelta))
    cites = [e.citation for e in second if isinstance(e, CitationEvent)]
    assert second[-1].reason == "end_turn", answer
    assert "5" in answer, answer
    assert cites, f"no [n] citation parsed from: {answer!r}"
    assert {c.chunk_id for c in cites} == {"local_files:handbook/pto.md#2"}


def test_anthropic_adapter_parallel_tool_use_via_ollama_messages_api():
    llm = AnthropicProvider(model=MODEL, api_key="unused", base_url=OLLAMA, max_tokens=2048)

    events = collect(llm.stream(SYSTEM, [{"role": "user", "content": QUESTION}], TOOLS))

    calls = [e for e in events if isinstance(e, ToolCall)]
    assert sorted(c.name for c in calls) == ["find_experts", "search_kb"]
    assert events[-1].reason == "tool_use"
    assert [b["type"] for b in events[-1].content if b["type"] == "tool_use"] == ["tool_use"] * 2
