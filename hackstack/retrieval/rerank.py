"""Rerankers: re-score the fused shortlist with a cross-encoder and keep the best ``top_n``.

- ``noop``: keep the fused (RRF) order. The instant cut, and the fallback when the optional
  ``rerank`` extra is not installed.
- ``cross_encoder``: a local sentence-transformers ``CrossEncoder``, in-process. The local
  profile uses ``cross-encoder/ms-marco-MiniLM-L6-v2`` on CPU, about 1.2 s for 30 candidates
  (docs/research/local-models.md). Ollama has no rerank endpoint, so this cannot go through it.
  Needs ``uv sync --extra rerank``.
- ``voyage``: Voyage ``rerank-3`` via the REST API. Tested against fixtures only: no live key.

Each candidate is scored as ``title + "\\n" + text`` (just ``text`` when there is no title).
Returned hits are copies with the reranker's ``score`` and ``retriever="rerank"``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import httpx

from hackstack import registry
from hackstack._http import post_json
from hackstack.embed.voyage import DEFAULT_BASE_URL as VOYAGE_BASE_URL
from hackstack.embed.voyage import require_voyage_key
from hackstack.models import SearchHit
from hackstack.registry import RERANKER, register
from hackstack.retrieval.base import Reranker

if TYPE_CHECKING:
    from hackstack.config import Secrets, Settings

Scorer = Callable[[list[tuple[str, str]]], Sequence[float]]

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"


class RerankerUnavailable(RuntimeError):
    """The configured reranker cannot run here (e.g. its optional dependencies are missing)."""


def passage(hit: SearchHit) -> str:
    return f"{hit.title}\n{hit.text}" if hit.title else hit.text


def _rescored(
    hits: list[SearchHit], scored: list[tuple[int, float]], top_n: int
) -> list[SearchHit]:
    best = sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_n]
    return [hits[i].model_copy(update={"score": float(s), "retriever": "rerank"}) for i, s in best]


@register(RERANKER, "noop")
class NoopReranker:
    """Keep the fused order; just cut to ``top_n``."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return list(hits[: max(top_n, 0)])


@register(RERANKER, "cross_encoder")
class CrossEncoderReranker:
    """A sentence-transformers cross-encoder, loaded lazily on first use and then kept."""

    def __init__(
        self,
        model: str = DEFAULT_CROSS_ENCODER,
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
        scorer: Scorer | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._scorer = scorer
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls, settings: Settings, secrets: Secrets | None = None
    ) -> CrossEncoderReranker:
        return cls(settings.rerank.model, device=settings.rerank.device)

    def load(self) -> Scorer:
        """Load the model now (otherwise the first ``rerank`` does). Useful as a warm-up."""
        with self._lock:
            if self._scorer is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RerankerUnavailable(
                        "rerank.provider is cross_encoder but sentence-transformers is not "
                        "installed: run `uv sync --extra rerank`, or set rerank.provider: noop"
                    ) from exc
                encoder = CrossEncoder(self.model, device=self.device, max_length=self.max_length)

                def score(pairs: list[tuple[str, str]]) -> Sequence[float]:
                    return encoder.predict(
                        pairs, batch_size=self.batch_size, show_progress_bar=False
                    )

                self._scorer = score
            return self._scorer

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits or top_n <= 0:
            return []
        scores = self.load()([(query, passage(h)) for h in hits])
        if len(scores) != len(hits):
            raise ValueError(f"{self.model} returned {len(scores)} scores for {len(hits)} hits")
        return _rescored(hits, list(enumerate(scores)), top_n)


@register(RERANKER, "voyage")
class VoyageReranker:
    """Voyage's hosted reranker (``rerank-3``). Billed per (query tokens x docs) + doc tokens."""

    def __init__(
        self,
        model: str = "rerank-3",
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = (base_url or VOYAGE_BASE_URL).rstrip("/")
        self._headers = {"Authorization": f"Bearer {require_voyage_key(api_key)}"}
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings, secrets: Secrets | None = None) -> VoyageReranker:
        key = secrets.voyage_api_key if secrets else None
        return cls(settings.rerank.model, api_key=key.get_secret_value() if key else None)

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits or top_n <= 0:
            return []
        body = post_json(
            self._client,
            f"{self.base_url}/rerank",
            {
                "model": self.model,
                "query": query,
                "documents": [passage(h) for h in hits],
                "top_k": top_n,
                "truncation": True,
            },
            provider="voyage",
            headers=self._headers,
            sleep=self._sleep,
        )
        scored = [(d["index"], d["relevance_score"]) for d in body["data"]]
        return _rescored(hits, scored, top_n)

    def close(self) -> None:
        self._client.close()


def build_reranker(settings: Settings, secrets: Secrets | None = None) -> Reranker:
    """The reranker ``rerank.provider`` names, configured from ``settings.rerank``."""
    from hackstack.config import load_secrets

    cls = registry.get(RERANKER, settings.rerank.provider)
    if hasattr(cls, "from_settings"):
        return cls.from_settings(settings, secrets if secrets is not None else load_secrets())
    return cls(model=settings.rerank.model)
