from __future__ import annotations

from typing import Protocol, runtime_checkable

from hackstack.models import SearchHit


@runtime_checkable
class Reranker(Protocol):
    """Register implementations with ``@register("reranker", "<name>")``."""

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...
