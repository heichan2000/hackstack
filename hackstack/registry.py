"""Name-based component registry.

Every swappable component (connector, parser, chunker, embedder, store, reranker,
LLM provider) registers itself under a ``kind`` and a config ``name``::

    @register("embedder", "voyage")
    class VoyageEmbedder: ...

and is looked up from ``hackstack.yaml`` with ``get("embedder", settings.embed.provider)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

# Kinds used by the kit. Other kinds are allowed; these are the seams in README.
CONNECTOR = "connector"
PARSER = "parser"
CHUNKER = "chunker"
EMBEDDER = "embedder"
STORE = "store"
RERANKER = "reranker"
LLM = "llm"

_registry: dict[str, dict[str, Any]] = {}


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class (or factory) decorator that registers ``obj`` as ``kind``/``name``."""

    def decorator(obj: T) -> T:
        names = _registry.setdefault(kind, {})
        if name in names and names[name] is not obj:
            raise ValueError(f"{kind} {name!r} is already registered by {names[name]!r}")
        names[name] = obj
        return obj

    return decorator


def get(kind: str, name: str) -> Any:
    """Return the component registered as ``kind``/``name``."""
    try:
        return _registry[kind][name]
    except KeyError:
        known = ", ".join(available(kind)) or "none"
        raise KeyError(f"no {kind} registered as {name!r}; available: {known}") from None


def available(kind: str) -> list[str]:
    """Sorted names registered for ``kind``."""
    return sorted(_registry.get(kind, {}))


def unregister_kind(kind: str) -> None:
    """Forget every registration for ``kind`` (for tests)."""
    _registry.pop(kind, None)
