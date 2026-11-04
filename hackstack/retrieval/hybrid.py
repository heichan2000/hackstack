"""Hybrid retrieval: dense + lexical under one ACL filter, fused by rank, reranked, expanded.

One ``search()`` call runs::

    embed query -> store.dense(top window) ─┐
                   store.lexical(top window) ┴> rrf(k, weights) -> rerank(shortlist) -> expand

Both retrievers receive the same ``principals`` and ``filters`` and the store applies them
inside its SQL (``store/postgres.py: filter_clause``), so neither side can see more than the
other. Reciprocal Rank Fusion uses ranks only, so BM25-ish ``ts_rank_cd`` values and cosine
similarities never need to be put on one scale.

Knobs (``hackstack.yaml``): ``retrieval.window`` candidates per retriever, ``rrf.k`` and
``rrf.weights`` (dense, lexical; a weight of 0 skips that retriever), ``rerank.window`` fused
candidates handed to the reranker, ``rerank.top_n`` hits kept, ``expand.*`` neighbours.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import SpanKind

from hackstack import semconv as sc
from hackstack import tracing
from hackstack.models import SearchHit

if TYPE_CHECKING:
    from hackstack.config import Settings
    from hackstack.embed.base import Embedder
    from hackstack.retrieval.base import Reranker
    from hackstack.retrieval.expand import Expander
    from hackstack.store.base import Store


def rrf(
    *ranked_lists: Sequence[str], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: ``score(d) = Σ_r w_r / (k + rank_r(d))``, best first.

    ``k=60`` is Cormack et al. (SIGIR 2009); lower k favours each list's top hit. An id
    repeated within one list counts once, at its best rank. Ties keep first-appearance order,
    so results are deterministic. Ids that only appear in zero-weight lists are dropped.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")
    if weights is not None:
        if len(weights) != len(ranked_lists):
            raise ValueError(f"rrf got {len(weights)} weights for {len(ranked_lists)} ranked lists")
        if any(w < 0 for w in weights):
            raise ValueError(f"rrf weights must be >= 0, got {list(weights)}")
    scores: dict[str, float] = {}
    for i, ranked in enumerate(ranked_lists):
        w = weights[i] if weights is not None else 1.0
        seen: set[str] = set()
        for rank, cid in enumerate(ranked, start=1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    # sorted() is stable: equal scores keep the order ids were first seen in.
    return sorted(
        ((cid, s) for cid, s in scores.items() if s > 0), key=lambda kv: kv[1], reverse=True
    )


def fuse(
    *hit_lists: Sequence[SearchHit], k: int = 60, weights: Sequence[float] | None = None
) -> list[SearchHit]:
    """``rrf`` over hit lists: copies of the hits, scored by RRF, ``retriever="hybrid"``."""
    by_id: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for h in hits:
            by_id.setdefault(h.chunk_id, h)
    ranked = rrf(*[[h.chunk_id for h in hits] for hits in hit_lists], k=k, weights=weights)
    return [
        by_id[cid].model_copy(update={"score": score, "retriever": "hybrid"})
        for cid, score in ranked
    ]


@dataclass
class RetrievalResult:
    """Every stage of one retrieval, for tracing, evals and the sources panel."""

    query: str
    dense: list[SearchHit] = field(default_factory=list)
    lexical: list[SearchHit] = field(default_factory=list)
    fused: list[SearchHit] = field(default_factory=list)
    reranked: list[SearchHit] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)  # reranked, then expanded


class HybridRetriever:
    """Dense + lexical retrieval, RRF fusion, a reranked shortlist and context expansion."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker,
        *,
        expander: Expander | None = None,
        window: int = 50,
        k: int = 60,
        weights: Sequence[float] = (1.0, 1.0),
        rerank_window: int = 50,
        top_n: int = 8,
    ) -> None:
        if len(weights) != 2:
            raise ValueError(f"rrf weights are [dense, lexical]; got {list(weights)}")
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.expander = expander
        self.window = window
        self.k = k
        self.weights = (float(weights[0]), float(weights[1]))
        self.rerank_window = rerank_window
        self.top_n = top_n

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        store: Store | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> HybridRetriever:
        """Build from ``hackstack.yaml``; pass any component to reuse an existing instance."""
        from hackstack.embed import build_embedder
        from hackstack.ingest import build
        from hackstack.registry import STORE
        from hackstack.retrieval.expand import Expander
        from hackstack.retrieval.rerank import build_reranker

        store = store if store is not None else build(STORE, settings.store.provider, settings)
        ex = settings.expand
        return cls(
            store=store,
            embedder=embedder if embedder is not None else build_embedder(settings),
            reranker=reranker if reranker is not None else build_reranker(settings),
            expander=(
                Expander(
                    store, radius=ex.radius, same_section=ex.same_section, max_chars=ex.max_chars
                )
                if ex.radius > 0
                else None
            ),
            window=settings.retrieval.window,
            k=settings.rrf.k,
            weights=tuple(settings.rrf.weights),
            rerank_window=settings.rerank.window,
            top_n=settings.rerank.top_n,
        )

    def retrieve(
        self,
        query: str,
        principals: list[str],
        filters: dict[str, Any] | None = None,
        *,
        top_n: int | None = None,
    ) -> RetrievalResult:
        """Run every stage and keep each one's output (each stage is a span when tracing)."""
        n = self.top_n if top_n is None else top_n
        attrs = {
            sc.OPERATION_NAME: sc.OP_RETRIEVAL,
            sc.OI_SPAN_KIND: sc.KIND_RETRIEVER,
            sc.DATA_SOURCE_ID: getattr(self.store, "name", None) or type(self.store).__name__,
            sc.HS_PRINCIPALS: len(principals),
            sc.HS_FILTERS: tuple(sorted(k for k, v in (filters or {}).items() if v)),
            sc.HS_TOP_N: n,
        }
        with tracing.span(sc.SPAN_RETRIEVAL, attrs) as span:
            tracing.content(span, sc.RETRIEVAL_QUERY_TEXT, query)
            result = self._retrieve(query, principals, filters, n)
            span.set_attribute(sc.RETRIEVAL_DOCUMENTS, tracing.documents(result.hits))
            span.set_attribute(sc.HS_OUTPUT_COUNT, len(result.hits))
            return result

    def _retrieve(
        self, query: str, principals: list[str], filters: dict[str, Any] | None, n: int
    ) -> RetrievalResult:
        result = RetrievalResult(query=query)
        if not principals or not query.strip():
            return result  # fail closed, and do not pay for an embedding
        w_dense, w_lexical = self.weights
        if w_dense > 0:
            model = getattr(self.embedder, "model", None)
            with tracing.span(
                f"{sc.SPAN_EMBEDDINGS} {model}" if model else sc.SPAN_EMBEDDINGS,
                {
                    sc.OPERATION_NAME: sc.OP_EMBEDDINGS,
                    sc.OI_SPAN_KIND: sc.KIND_EMBEDDING,
                    sc.REQUEST_MODEL: model,
                    **tracing.server_attributes(getattr(self.embedder, "base_url", None)),
                },
                kind=SpanKind.CLIENT,
            ) as span:
                qvec = self.embedder.embed_query(query)
                span.set_attribute(sc.EMBEDDING_DIMENSIONS, len(qvec))
            with _stage(sc.SPAN_DENSE, {sc.HS_WINDOW: self.window}) as span:
                result.dense = self.store.dense(qvec, principals, filters, self.window)
                span.set_attribute(sc.HS_OUTPUT_COUNT, len(result.dense))
        if w_lexical > 0:
            with _stage(sc.SPAN_LEXICAL, {sc.HS_WINDOW: self.window}) as span:
                result.lexical = self.store.lexical(query, principals, filters, self.window)
                span.set_attribute(sc.HS_OUTPUT_COUNT, len(result.lexical))
        with _stage(
            sc.SPAN_RRF,
            {sc.HS_RRF_K: self.k, sc.HS_INPUT_COUNT: len(result.dense) + len(result.lexical)},
        ) as span:
            result.fused = fuse(result.dense, result.lexical, k=self.k, weights=self.weights)
            span.set_attribute(sc.HS_OUTPUT_COUNT, len(result.fused))
        shortlist = result.fused[: self.rerank_window]
        with _stage(
            sc.SPAN_RERANK,
            {
                sc.OI_SPAN_KIND: sc.KIND_RERANKER,
                sc.HS_RERANK_MODEL: getattr(self.reranker, "model", None),
                sc.HS_INPUT_COUNT: len(shortlist),
                sc.HS_TOP_N: n,
            },
        ) as span:
            result.reranked = self.reranker.rerank(query, shortlist, n)
            span.set_attribute(sc.HS_OUTPUT_COUNT, len(result.reranked))
        if self.expander is not None:
            with _stage(
                sc.SPAN_EXPAND,
                {
                    sc.HS_EXPAND_RADIUS: getattr(self.expander, "radius", None),
                    sc.HS_INPUT_COUNT: len(result.reranked),
                },
            ) as span:
                result.hits = self.expander.expand(result.reranked, principals)
                span.set_attribute(sc.HS_OUTPUT_COUNT, len(result.hits))
        else:
            result.hits = list(result.reranked)
        return result

    def search(
        self,
        query: str,
        principals: list[str],
        filters: dict[str, Any] | None = None,
        *,
        top_n: int | None = None,
    ) -> list[SearchHit]:
        """The final hits: fused, reranked and expanded, all under ``principals``."""
        return self.retrieve(query, principals, filters, top_n=top_n).hits


def _stage(name: str, attrs: dict[str, Any]) -> Any:
    """A retrieval stage span (a ``CHAIN`` unless ``attrs`` say otherwise)."""
    return tracing.span(name, {sc.OI_SPAN_KIND: sc.KIND_CHAIN, **attrs})
