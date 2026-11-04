"""Embedding models (asymmetric: documents vs. queries).

``build_embedder(settings)`` returns the one named by ``embed.provider``: ``ollama``,
``voyage`` or ``openai_compat`` (or anything else registered as an ``embedder``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hackstack import registry
from hackstack.embed import ollama as _ollama  # noqa: F401  registers "ollama"
from hackstack.embed import openai_compat as _openai_compat  # noqa: F401  "openai_compat"
from hackstack.embed import voyage as _voyage  # noqa: F401  registers "voyage"
from hackstack.embed.base import Embedder

if TYPE_CHECKING:
    from hackstack.config import Secrets, Settings


def build_embedder(settings: Settings, secrets: Secrets | None = None) -> Embedder:
    """The embedder ``embed.provider`` names, configured from ``settings.embed``."""
    from hackstack.config import load_secrets

    cls = registry.get(registry.EMBEDDER, settings.embed.provider)
    if hasattr(cls, "from_settings"):
        return cls.from_settings(settings, secrets if secrets is not None else load_secrets())
    return cls(model=settings.embed.model, dim=settings.embed.dim)
