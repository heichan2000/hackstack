"""Embedder for any OpenAI-compatible ``/embeddings`` endpoint (OpenAI, vLLM, Ollama's ``/v1``).

``dimensions`` is not sent, because many compatible servers reject it; vectors wider than
``embed.dim`` are truncated and re-normalized locally instead, which for Matryoshka models is the
same thing. ``OPENAI_API_KEY`` is sent only when set (Ollama and vLLM need none).
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


@register(EMBEDDER, "openai_compat")
class OpenAICompatEmbedder:
    def __init__(
        self,
        model: str,
        dim: int,
        *,
        base_url: str | None,
        api_key: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 64,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url:
            raise ValueError(
                "embed.base_url is required for openai_compat, e.g. https://api.openai.com/v1"
            )
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    @classmethod
    def from_settings(
        cls, settings: Settings, secrets: Secrets | None = None
    ) -> OpenAICompatEmbedder:
        key = secrets.openai_api_key if secrets else None
        e = settings.embed
        return cls(
            e.model,
            e.dim,
            base_url=e.base_url,
            api_key=key.get_secret_value() if key else None,
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
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": inputs},
            provider="openai_compat",
            headers=self._headers,
            sleep=self._sleep,
        )
        data = sorted(body["data"], key=lambda d: d["index"])
        if len(data) != len(inputs):
            raise ValueError(
                f"{self.base_url} returned {len(data)} embeddings for {len(inputs)} inputs"
            )
        return [fit_dim(d["embedding"], self.dim, model=self.model) for d in data]
