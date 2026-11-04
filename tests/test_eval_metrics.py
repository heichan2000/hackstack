"""Retrieval metrics against hand-computed examples (binary relevance)."""

import math

import pytest

from evals.metrics import (
    hit_rate_at_k,
    mean_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "x"}  # x is never retrieved


def test_recall_is_relevant_retrieved_over_all_relevant():
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0


def test_precision_divides_by_k_even_when_fewer_were_returned():
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(2 / 10)


def test_reciprocal_rank_is_one_over_the_first_relevant_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1 / 2)
    assert reciprocal_rank(["e", "b"], RELEVANT) == 1.0
    assert reciprocal_rank(["a", "c"], RELEVANT) == 0.0


def test_reciprocal_rank_respects_a_cutoff():
    assert reciprocal_rank(RANKED, {"e"}, k=4) == 0.0
    assert reciprocal_rank(RANKED, {"e"}, k=5) == pytest.approx(1 / 5)


def test_hit_rate_is_any_relevant_in_the_top_k():
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0


def test_ndcg_by_hand():
    # DCG = 1/log2(3) + 1/log2(6); ideal puts all 3 relevant first: 1 + 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)  # ~0.4776
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.47763, abs=1e-5)


def test_ndcg_ideal_is_capped_at_k():
    # One slot, three relevant: a relevant top-1 is already ideal.
    assert ndcg_at_k(["b", "a"], RELEVANT, 1) == 1.0


def test_perfect_ranking_scores_one_everywhere():
    ranked, relevant = ["p", "q"], {"p", "q"}
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert reciprocal_rank(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(1.0)


def test_duplicates_in_a_ranking_count_once():
    # Context expansion or a buggy fusion could repeat an id; it must not inflate the score.
    assert recall_at_k(["b", "b", "b"], {"b", "e"}, 3) == pytest.approx(1 / 2)
    assert precision_at_k(["b", "b", "c"], {"b"}, 3) == pytest.approx(1 / 3)
    dcg = 1.0  # b at rank 1 only; the repeats are dropped, so c moves up to rank 2
    assert ndcg_at_k(["b", "b", "c"], {"b"}, 3) == pytest.approx(dcg / 1.0)


def test_no_relevant_labels_is_an_error_not_a_zero():
    # Unanswerable questions are scored separately; averaging them in as 0 would hide that.
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, set(), 5)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, RELEVANT, 0)


def test_mean_metrics_averages_each_metric_over_queries():
    runs = [
        (["a", "b"], {"a"}),  # perfect
        (["c", "d"], {"x"}),  # miss
    ]
    m = mean_metrics(runs, ks=(1, 2))
    assert m["recall@1"] == pytest.approx(0.5)
    assert m["recall@2"] == pytest.approx(0.5)
    assert m["precision@1"] == pytest.approx(0.5)
    assert m["precision@2"] == pytest.approx(0.25)
    assert m["hit@1"] == pytest.approx(0.5)
    assert m["mrr"] == pytest.approx(0.5)
    assert m["ndcg@2"] == pytest.approx(0.5)
    assert m["n"] == 2


def test_mean_metrics_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        mean_metrics([], ks=(5,))
