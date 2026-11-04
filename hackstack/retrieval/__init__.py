"""Hybrid retrieval: dense + lexical, RRF fusion, rerank, context expansion."""

from hackstack.retrieval import rerank as _rerank  # noqa: F401  registers the rerankers
