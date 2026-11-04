"""Reciprocal Rank Fusion: ``rrf()`` over id lists and ``fuse()`` over SearchHits."""

from __future__ import annotations

import pytest

from hackstack.models import SearchHit
from hackstack.retrieval.hybrid import fuse, rrf


def test_rrf_scores_are_the_sum_of_reciprocal_ranks():
    fused = dict(rrf(["a", "b"], ["b", "c"], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_ranks_agreement_above_a_single_top_hit():
    # "b" is 2nd in both lists; "a" is 1st in one and absent from the other.
    order = [cid for cid, _ in rrf(["a", "b", "c"], ["d", "b", "e"])]
    assert order[0] == "b"


def test_single_retriever_hits_are_kept_but_rank_below_shared_ones():
    order = [cid for cid, _ in rrf(["shared", "dense_only"], ["shared", "lexical_only"])]
    assert order[0] == "shared"
    assert set(order) == {"shared", "dense_only", "lexical_only"}


def test_one_list_keeps_its_order():
    assert [cid for cid, _ in rrf(["x", "y", "z"])] == ["x", "y", "z"]


def test_weights_scale_each_list():
    fused = dict(rrf(["a"], ["b"], k=60, weights=[2.0, 0.5]))
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(0.5 / 61)
    order = [cid for cid, _ in rrf(["a"], ["b"], weights=[0.5, 2.0])]
    assert order == ["b", "a"]


def test_zero_weight_drops_a_lists_contribution():
    order = [cid for cid, _ in rrf(["a", "b"], ["b", "c"], weights=[1.0, 0.0])]
    assert order == ["a", "b"]  # "c" scored 0 and is not returned


def test_lower_k_favours_the_top_rank():
    # a: rank 1 in one list. b: rank 3 in both. Low k rewards a's top-1, high k b's agreement.
    lists = (["a", "x", "b"], ["y", "z", "b"])
    assert rrf(*lists, k=1)[0][0] == "a"
    assert rrf(*lists, k=60)[0][0] == "b"


def test_ties_break_by_first_appearance_deterministically():
    # a and b both score 1/61 (rank 1 in one list each): list order decides.
    assert [cid for cid, _ in rrf(["a"], ["b"])] == ["a", "b"]
    assert [cid for cid, _ in rrf(["b"], ["a"])] == ["b", "a"]


def test_duplicate_ids_within_a_list_count_once_at_their_best_rank():
    fused = dict(rrf(["a", "a", "b"]))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)


def test_rrf_validates_inputs():
    with pytest.raises(ValueError, match="weights"):
        rrf(["a"], ["b"], weights=[1.0])
    with pytest.raises(ValueError, match="k"):
        rrf(["a"], k=0)
    with pytest.raises(ValueError, match="weights"):
        rrf(["a"], weights=[-1.0])
    assert rrf() == []
    assert rrf([], []) == []


def hit(cid, retriever, score=1.0, **kw):
    return SearchHit(
        chunk_id=cid,
        doc_id=cid.split("#")[0],
        text=f"text {cid}",
        score=score,
        retriever=retriever,
        **kw,
    )


def test_fuse_returns_hybrid_hits_scored_by_rrf():
    dense = [hit("a#0", "dense", 0.9), hit("b#0", "dense", 0.8)]
    lexical = [hit("b#0", "lexical", 7.0), hit("c#0", "lexical", 3.0)]
    fused = fuse(dense, lexical, k=60)
    assert [h.chunk_id for h in fused] == ["b#0", "a#0", "c#0"]
    assert all(h.retriever == "hybrid" for h in fused)
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    # The hit's other fields survive fusion.
    assert fused[1].text == "text a#0" and fused[1].doc_id == "a"


def test_fuse_does_not_mutate_its_inputs():
    dense = [hit("a#0", "dense", 0.9)]
    fuse(dense, [], k=60)
    assert dense[0].retriever == "dense" and dense[0].score == 0.9
