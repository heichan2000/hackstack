"""Replay recorded server-sent-event streams through an httpx2 mock transport.

``tests/fixtures/llm/*.sse`` are raw HTTP response bodies. The ``openai_*`` and
``anthropic_compat_*`` files were recorded from the local Ollama (``hackstack-chat``); the
``claude_*`` files are hand-written from the Messages API streaming docs, because no Anthropic
key was available to record them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2

FIXTURES = Path(__file__).parent / "fixtures" / "llm"


class Replay:
    """A mock transport that answers every request with the next fixture, and records bodies."""

    def __init__(self, *fixtures: str | bytes, status: int = 200) -> None:
        """Each fixture is a file name under ``fixtures/llm`` or a literal response body."""
        self._bodies = [
            f if isinstance(f, bytes) else (FIXTURES / f).read_bytes() for f in fixtures
        ]
        self._status = status
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx2.Headers] = []
        self.transport = httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content or b"{}"))
        self.headers.append(request.headers)
        body = self._bodies.pop(0)
        return httpx2.Response(
            self._status, content=body, headers={"content-type": "text/event-stream"}
        )

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=self.transport)

    @property
    def body(self) -> dict[str, Any]:
        """The JSON body of the only request made."""
        assert len(self.requests) == 1, self.requests
        return self.requests[0]


def collect(stream: AsyncIterator[Any]) -> list[Any]:
    """Drain an async event stream into a list."""

    async def _drain() -> list[Any]:
        return [event async for event in stream]

    return asyncio.run(_drain())
