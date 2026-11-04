from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from hackstack.models import LLMEvent


class LLMError(RuntimeError):
    """A provider call failed (HTTP error, unreachable server, unknown model ...)."""


@runtime_checkable
class LLMProvider(Protocol):
    """Register implementations with ``@register("llm", "<name>")``.

    ``stream`` yields normalized events (text deltas, tool calls, citations) and always ends
    with exactly one ``Stop``. ``messages`` and ``tools`` use the Anthropic Messages shape;
    adapters for other APIs translate.
    """

    def stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **opts: Any,
    ) -> AsyncIterator[LLMEvent]: ...
