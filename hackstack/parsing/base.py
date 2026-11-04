from __future__ import annotations

from typing import Protocol, runtime_checkable

from hackstack.models import ParsedDoc


@runtime_checkable
class Parser(Protocol):
    """Register implementations with ``@register("parser", "<name>")``."""

    def parse(self, content: bytes | str, mime: str) -> ParsedDoc: ...
