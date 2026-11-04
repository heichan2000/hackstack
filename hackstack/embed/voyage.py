"""Voyage embedder (``voyage-4`` at 1024 dims by default), via the REST API.

Asymmetry comes from ``input_type`` (``document`` / ``query``); the dimension from
``output_dimension`` (voyage-4 supports 256, 512, 1024 and 2048).

Tested against fixtures shaped like Voyage's API reference only: no live key yet.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

from hackstack._http import post_json
from hackstack.embed.base import batched, fit_dim
from hackstack.registry import EMBEDDER, register

if TYPE_CHECKING:
    from hackstack.config import Secrets, Settings

DEFAULT_BASE_URL = "https://api.voyageai.com/v1"


def require_voyage_key(api_key: str | None) -> str:
    if not api_key:
        raise ValueError(
            "VOYAGE_API_KEY is not set: add it to .env, or run without paid APIs with "
            "HACKSTACK_PROFILE=local"
        )
    return api_key


@register(EMBEDDER, "voyage")
class VoyageEmbedder:
    def __init__(
        self,
        model: str = "voyage-4",
        dim: int = 1024,
        *,
        api_key: str | None,
        base_url: str | None = None,
        batch_size: int = 128,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"Authorization": f"Bearer {require_voyage_key(api_key)}"}
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings, secrets: Secrets | None = None) -> VoyageEmbedder:
        key = secrets.voyage_api_key if secrets else None
        e = settings.embed
        return cls(
            e.model,
            e.dim,
            api_key=key.get_secret_value() if key else None,
            base_url=e.base_url,
            batch_size=e.batch_size,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in batched(texts, self.batch_size):
            out.extend(self._embed(list(batch), "document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def close(self) -> None:
        self._client.close()

    def _embed(self, inputs: list[str], input_type: str) -> list[list[float]]:
        body = post_json(
            self._client,
            f"{self.base_url}/embeddings",
            {
                "model": self.model,
                "input": inputs,
                "input_type": input_type,
                "output_dimension": self.dim,
            },
            provider="voyage",
            headers=self._headers,
            sleep=self._sleep,
        )
        data = sorted(body["data"], key=lambda d: d["index"])
        if len(data) != len(inputs):
            raise ValueError(f"voyage returned {len(data)} embeddings for {len(inputs)} inputs")
        return [fit_dim(d["embedding"], self.dim, model=self.model) for d in data]
