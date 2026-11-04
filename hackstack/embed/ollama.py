"""Ollama embedder, via the native ``/api/embed`` endpoint.

The local profile uses ``qwen3-embedding:0.6b`` (1024 dims native). It is instruction-aware:
queries are sent as ``Instruct: <task>\\nQuery: <q>`` and documents unprefixed
(``embed.query_instruction``). See docs/research/local-models.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

from hackstack._http import post_json
from hackstack.embed.base import batched, fit_dim, qwen_query
from hackstack.registry import EMBEDDER, register

if TYPE_CHECKING:
    from hackstack.config import Secrets, Settings

DEFAULT_BASE_URL = "http://localhost:11434"


@register(EMBEDDER, "ollama")
class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        dim: int,
        *,
        base_url: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 32,
        timeout: float = 120.0,  # the first call may load the model from disk
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.dim = dim
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        # embed.base_url may be copied from llm.base_url (".../v1"); the native API is at the root
        self.base_url = url.removesuffix("/v1")
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings, secrets: Secrets | None = None) -> OllamaEmbedder:
        e = settings.embed
        return cls(
            e.model,
            e.dim,
            base_url=e.base_url,
            query_instruction=e.query_instruction,
            batch_size=e.batch_size,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in batched(texts, self.batch_size):
            out.extend(self._embed(list(batch)))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([qwen_query(self.query_instruction, text)])[0]

    def close(self) -> None:
        self._client.close()

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        body = post_json(
            self._client,
            f"{self.base_url}/api/embed",
            {"model": self.model, "input": inputs},
            provider="ollama",
            hint=self._hint,
            sleep=self._sleep,
        )
        vectors = body["embeddings"]
        if len(vectors) != len(inputs):
            raise ValueError(f"ollama returned {len(vectors)} embeddings for {len(inputs)} inputs")
        return [fit_dim(v, self.dim, model=self.model) for v in vectors]

    def _hint(self, status: int | None, detail: str) -> str:
        if status is None:
            return (
                "Is Ollama running? Start it (`ollama serve`) "
                f"or set embed.base_url (now {self.base_url})."
            )
        if status == 404 or "not found" in detail:
            return f"Pull the model first: ollama pull {self.model}"
        return ""
