from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Embedder(Protocol):
    """Register implementations with ``@register("embedder", "<name>")``.

    ``model`` and ``dim`` are stored with the index; mixing vectors from different models is
    refused, so changing the embedder means ``make reset`` and a re-ingest.

    An implementation may offer ``from_settings(settings, secrets)``; otherwise
    :func:`hackstack.embed.build_embedder` calls it with ``model=`` and ``dim=``.
    """

    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def fit_dim(vec: Sequence[float], dim: int, *, model: str) -> list[float]:
    """Return ``vec`` at exactly ``dim`` dimensions.

    Wider vectors are truncated to their first ``dim`` dimensions and re-normalized, which is
    how Matryoshka-trained models (voyage-4, qwen3-embedding, OpenAI v3) are meant to be
    shortened. Narrower vectors cannot be fixed, so they raise.
    """
    if len(vec) == dim:
        return list(vec)
    if len(vec) < dim:
        raise ValueError(
            f"{model} returned {len(vec)} dims but embed.dim is {dim}: set embed.dim to "
            f"{len(vec)} or less (then `make reset` and re-ingest)"
        )
    head = vec[:dim]
    norm = math.sqrt(sum(x * x for x in head))
    return [x / norm for x in head] if norm else list(head)


def qwen_query(instruction: str | None, text: str) -> str:
    """Format a query for instruction-aware embedders (Qwen3-Embedding style).

    Documents are embedded as-is; only queries carry the task instruction.
    """
    return f"Instruct: {instruction}\nQuery: {text}" if instruction else text


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size < 1:
        raise ValueError(f"batch size must be at least 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]
