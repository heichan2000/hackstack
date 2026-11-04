from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Register implementations with ``@register("embedder", "<name>")``.

    ``model`` and ``dim`` are stored with the index; mixing vectors from different models is
    refused, so changing the embedder means ``make reset`` and a re-ingest.
    """

    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
