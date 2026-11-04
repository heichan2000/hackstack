"""Deterministic fake models so ``make test`` needs no keys, network or GPU.

Both register under the config name ``fake``, so a test can set ``embed.provider: fake`` /
``llm.provider: fake`` and exercise the real wiring.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from hackstack.models import LLMEvent, Stop, TextDelta, ToolCall
from hackstack.registry import EMBEDDER, LLM, register

_TOKEN = re.compile(r"\w+")


@register(EMBEDDER, "fake")
class FakeEmbedder:
    """Feature-hashed bag of words: same text -> same unit vector; shared words -> closer."""

    def __init__(self, model: str = "fake-embed", dim: int = 1024) -> None:
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec


@register(LLM, "fake")
class FakeLLM:
    """Plays a script of turns, one per ``stream`` call; unscripted, it echoes the user.

    A turn without a ``Stop`` gets one appended: ``tool_use`` if it made tool calls, else
    ``end_turn``. Every call's arguments are recorded in ``calls``.
    """

    def __init__(self, script: Sequence[Sequence[LLMEvent]] | None = None) -> None:
        self._script = [list(turn) for turn in script] if script is not None else None
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **opts: Any,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append(
            copy.deepcopy({"system": system, "messages": messages, "tools": tools, "opts": opts})
        )
        for event in self._next_turn(messages):
            yield event

    def _next_turn(self, messages: list[dict[str, Any]]) -> list[LLMEvent]:
        if self._script is None:
            return [TextDelta(text=f"echo: {_last_user_text(messages)}"), Stop(reason="end_turn")]
        if not self._script:
            raise AssertionError(f"FakeLLM script exhausted after {len(self.calls) - 1} turns")
        turn = self._script.pop(0)
        if not any(isinstance(e, Stop) for e in turn):
            made_calls = any(isinstance(e, ToolCall) for e in turn)
            turn.append(Stop(reason="tool_use" if made_calls else "end_turn"))
        return turn


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return " ".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""
