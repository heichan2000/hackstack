"""LLM provider adapters, normalized to a stream of LLMEvents.

``make_llm(settings)`` builds the provider ``llm.provider`` names: ``anthropic``
(:mod:`.anthropic_provider`) or ``openai_compat`` (:mod:`.openai_compat_provider`).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from hackstack.registry import LLM, available, get

if TYPE_CHECKING:
    from hackstack.config import Secrets, Settings
    from hackstack.llm.base import LLMProvider

_BUILTIN = {
    "anthropic": "hackstack.llm.anthropic_provider",
    "openai_compat": "hackstack.llm.openai_compat_provider",
}


def make_llm(
    settings: Settings, secrets: Secrets | None = None, *, aux: bool = False
) -> LLMProvider:
    """The configured provider, for answers (``llm.answer_model``) or helper tasks (``aux``)."""
    from hackstack.config import load_secrets

    secrets = secrets if secrets is not None else load_secrets()
    name = settings.llm.provider
    if name in _BUILTIN and name not in available(LLM):
        importlib.import_module(_BUILTIN[name])  # registers it
    cls = get(LLM, name)
    model = settings.llm.aux_model if aux else settings.llm.answer_model
    if not hasattr(cls, "from_settings"):
        return cls()
    return cls.from_settings(settings.llm, secrets, model=model)
