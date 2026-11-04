"""Retrieval metrics, by hand: deterministic arithmetic over ranked ids and a relevant set.

Relevance is binary (a chunk answers the question or it does not). Every function takes the
ranking a retriever returned, best first, and the set of relevant ids. Repeated ids in a ranking
count once, at their best rank, so a stage that duplicates hits cannot inflate a score.

- ``recall@k``    relevant ids in the top k / all relevant ids. The one to watch: a generator
  can ignore noise, but it cannot recover a passage retrieval never handed it.
- ``precision@k`` relevant ids in the top k / k.
- ``MRR``         mean of 1 / rank of the first relevant id (0 when none is retrieved).
- ``hit@k``       1 if any relevant id is in the top k.
- ``nDCG@k``      DCG = Σ rel_i / log2(i + 1) over the top k, divided by the ideal DCG, which
  puts min(|relevant|, k) relevant ids first.

A question with no relevant ids (not in the corpus, or not visible to the asker) has no recall;
it raises instead of counting as 0, and is scored separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

DEFAULT_KS = (1, 3, 5, 10)


def _top(ranked: Sequence[str], k: int | None) -> list[str]:
    if k is not None and k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    seen: dict[str, None] = {}
    for rid in ranked:
        seen.setdefault(rid, None)
    unique = list(seen)
    return unique if k is None else unique[:k]


def _need_relevant(relevant: set[str]) -> None:
    if not relevant:
        raise ValueError("no relevant ids: score unanswerable questions separately")


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _need_relevant(relevant)
    return len(set(_top(ranked, k)) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _need_relevant(relevant)
    return len(set(_top(ranked, k)) & relevant) / k


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _need_relevant(relevant)
    return 1.0 if set(_top(ranked, k)) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    _need_relevant(relevant)
    for rank, rid in enumerate(_top(ranked, k), start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _need_relevant(relevant)
    dcg = sum(
        1.0 / math.log2(i + 1) for i, rid in enumerate(_top(ranked, k), start=1) if rid in relevant
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


def score_query(
    ranked: Sequence[str], relevant: set[str], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    """Every metric for one query, keyed ``recall@5``, ``mrr``, ``ndcg@10`` ..."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"hit@{k}"] = hit_rate_at_k(ranked, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    out["mrr"] = reciprocal_rank(ranked, relevant)
    return out


def mean_metrics(
    runs: Iterable[tuple[Sequence[str], set[str]]], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    """Mean of each metric over ``(ranked, relevant)`` pairs, plus ``n`` (the query count)."""
    per_query = [score_query(ranked, relevant, ks) for ranked, relevant in runs]
    if not per_query:
        raise ValueError("no queries to score")
    keys = per_query[0]
    out = {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}
    out["n"] = len(per_query)
    return out
