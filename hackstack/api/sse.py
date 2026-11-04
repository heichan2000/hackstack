"""The ``/chat`` SSE contract: agent events -> the kit's own event vocabulary.

Owning the schema (rather than a UI SDK's) keeps the browser and the server swappable. Every frame
is ``event: <type>`` plus one ``data:`` line of JSON whose ``type`` repeats the event name.
The vocabulary, documented in ``docs/sse.md``:

``text_delta``, ``citation``, ``tool_start``, ``tool_end``, ``grounding``, ``error``, ``done``.

A stream ends after exactly one ``done`` (preceded by ``grounding``) or one ``error``.
"""

from __future__ import annotations

import json
from typing import Any

from hackstack.agent.citations import CitationRegistry
from hackstack.agent.loop import AgentDone, AgentResult, ToolEnd, ToolStart
from hackstack.models import CitationEvent, TextDelta

EVENT_TYPES = (
    "text_delta",
    "citation",
    "tool_start",
    "tool_end",
    "grounding",
    "error",
    "done",
)


def frame(event_type: str, data: dict[str, Any]) -> str:
    """One SSE frame. ``data`` gets ``type`` set; JSON keeps it on a single line."""
    payload = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def comment(text: str = "") -> str:
    """An SSE comment line (ignored by clients); used to flush headers early."""
    return f": {text}\n\n"


class SSEEncoder:
    """Turns one turn's agent events into frames; tracks queries and citation numbers.

    ``history`` is what the model was sent, so citations to passages from earlier turns resolve
    to the same numbers the model used.
    """

    def __init__(self, conversation_id: str, history: list[dict[str, Any]] | None = None) -> None:
        self.conversation_id = conversation_id
        self._queries: dict[str, str | None] = {}
        self._numbers: dict[str, int] = {}
        for ref in CitationRegistry.from_messages(history or []).refs:
            self._numbers.setdefault(ref.chunk_id, ref.number)

    def encode(self, event: Any) -> list[str]:
        if isinstance(event, TextDelta):
            return [frame("text_delta", {"text": event.text})]
        if isinstance(event, CitationEvent):
            c = event.citation
            number = self._numbers.get(c.chunk_id or "")
            return [frame("citation", {"number": number, "citation": c.model_dump(mode="json")})]
        if isinstance(event, ToolStart):
            query = _query(event.input)
            self._queries[event.id] = query
            return [
                frame(
                    "tool_start",
                    {"id": event.id, "name": event.name, "input": event.input, "query": query},
                )
            ]
        if isinstance(event, ToolEnd):
            for hit in event.hits:
                self._numbers[hit.chunk_id] = hit.number  # the latest showing wins
            data = event.model_dump(mode="json", exclude={"type"})
            data["query"] = self._queries.get(event.id)
            return [frame("tool_end", data)]
        if isinstance(event, AgentDone):
            return [self.grounding(event.result), self.done(event.result)]
        return []

    def grounding(self, result: AgentResult) -> str:
        cited = sorted({s.number for s in result.sources if s.cited})
        return frame(
            "grounding",
            {
                "low_grounding": result.low_grounding,
                "cited": cited,
                "sources": [s.model_dump(mode="json") for s in result.sources],
            },
        )

    def done(self, result: AgentResult) -> str:
        data = result.model_dump(mode="json", exclude={"messages", "tool_calls", "sources"})
        data["conversation_id"] = self.conversation_id
        return frame("done", data)

    @staticmethod
    def error(message: str, code: str = "internal_error") -> str:
        return frame("error", {"code": code, "message": message})


def _query(args: dict[str, Any]) -> str | None:
    for key in ("query", "q", "topic", "doc", "document"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return None
