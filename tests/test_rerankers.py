"""Rerankers: no-op, local cross-encoder (with an injected scorer) and Voyage (mocked HTTP).

The Voyage response in ``tests/fixtures/voyage/rerank.json`` follows the shape in Voyage's API
reference (https://docs.voyageai.com/reference/reranker-api); it was not recorded live.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from hackstack import registry
from hackstack._http import ProviderError
from hackstack.config import Secrets, load_settings
from hackstack.models import SearchHit
from hackstack.retrieval.base import Reranker
from hackstack.retrieval.rerank import (
    CrossEncoderReranker,
    NoopReranker,
    RerankerUnavailable,
    VoyageReranker,
    build_reranker,
)

FIXTURE = Path(__file__).parent / "fixtures" / "voyage" / "rerank.json"


def hit(i, text, title="", score=0.0):
    return SearchHit(
        chunk_id=f"c{i}", doc_id=f"d{i}", text=text, title=title, score=score, retriever="hybrid"
    )


HITS = [
    hit(0, "PTO carries over up to five days.", title="PTO policy", score=0.03),
    hit(1, "The cafeteria opens at eight.", score=0.02),
    hit(2, "Carryover was raised to ten days in November.", title="PTO update", score=0.01),
]


# --- no-op -------------------------------------------------------------------------------------


def test_noop_is_registered_and_keeps_the_fused_order_cut_to_top_n():
    assert registry.get("reranker", "noop") is NoopReranker
    rr = NoopReranker()
    assert isinstance(rr, Reranker)
    assert rr.rerank("q", HITS, top_n=2) == HITS[:2]


def test_noop_does_not_mutate_its_input():
    hits = list(HITS)
    NoopReranker().rerank("q", hits, top_n=1)
    assert hits == HITS


# --- cross-encoder (scorer injected; the live model runs in test_live_models.py) ---------------


class RecordingScorer:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def __call__(self, pairs):
        self.pairs = pairs
        return self.scores


def test_cross_encoder_is_registered_and_satisfies_the_interface():
    assert registry.get("reranker", "cross_encoder") is CrossEncoderReranker
    assert isinstance(CrossEncoderReranker(scorer=RecordingScorer([])), Reranker)


def test_cross_encoder_sorts_by_score_keeps_top_n_and_marks_the_retriever():
    scorer = RecordingScorer([0.4, -3.0, 2.5])
    out = CrossEncoderReranker(scorer=scorer).rerank("how much PTO carries over?", HITS, top_n=2)

    assert [h.chunk_id for h in out] == ["c2", "c0"]
    assert [h.score for h in out] == [2.5, 0.4]
    assert all(h.retriever == "rerank" for h in out)
    # inputs are untouched: hits are copied, not mutated
    assert HITS[2].score == 0.01 and HITS[2].retriever == "hybrid"


def test_cross_encoder_pairs_the_query_with_title_and_text():
    scorer = RecordingScorer([0.0, 0.0, 0.0])
    CrossEncoderReranker(scorer=scorer).rerank("q", HITS, top_n=3)
    assert scorer.pairs == [
        ("q", "PTO policy\nPTO carries over up to five days."),
        ("q", "The cafeteria opens at eight."),
        ("q", "PTO update\nCarryover was raised to ten days in November."),
    ]


def test_cross_encoder_with_no_hits_or_zero_top_n_does_not_score():
    scorer = RecordingScorer([])
    rr = CrossEncoderReranker(scorer=scorer)
    assert rr.rerank("q", [], top_n=5) == []
    assert rr.rerank("q", HITS, top_n=0) == []
    assert scorer.pairs is None


def test_cross_encoder_without_the_optional_extra_explains_the_fix(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sentence_transformers)
    rr = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L6-v2")
    with pytest.raises(
        RerankerUnavailable, match=r"uv sync --extra rerank.*rerank\.provider: noop"
    ):
        rr.rerank("q", HITS, top_n=2)


# --- Voyage (fixture shaped like the API reference; not recorded live) -------------------------


def voyage_mock(response=None, status=200):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json=response or json.loads(FIXTURE.read_text()))

    return requests, httpx.Client(transport=httpx.MockTransport(handler))


def test_voyage_reranker_is_registered_and_satisfies_the_interface():
    assert registry.get("reranker", "voyage") is VoyageReranker
    assert isinstance(VoyageReranker(api_key="k", client=voyage_mock()[1]), Reranker)


def test_voyage_reranker_maps_indices_back_to_hits():
    requests, client = voyage_mock()
    out = VoyageReranker("rerank-3", api_key="vk", client=client).rerank("pto?", HITS, top_n=2)

    assert [h.chunk_id for h in out] == ["c2", "c0"]
    assert [h.score for h in out] == [0.912109375, 0.455078125]
    assert all(h.retriever == "rerank" for h in out)
    (request,) = requests
    assert str(request.url) == "https://api.voyageai.com/v1/rerank"
    assert request.headers["authorization"] == "Bearer vk"
    assert json.loads(request.content) == {
        "model": "rerank-3",
        "query": "pto?",
        "documents": [
            "PTO policy\nPTO carries over up to five days.",
            "The cafeteria opens at eight.",
            "PTO update\nCarryover was raised to ten days in November.",
        ],
        "top_k": 2,
        "truncation": True,
    }


def test_voyage_reranker_with_no_hits_makes_no_request():
    requests, client = voyage_mock()
    assert VoyageReranker(api_key="k", client=client).rerank("q", [], top_n=3) == []
    assert requests == []


def test_voyage_reranker_http_error_is_a_provider_error():
    _, client = voyage_mock({"detail": "bad"}, status=400)
    with pytest.raises(ProviderError, match="voyage: HTTP 400"):
        VoyageReranker(api_key="k", client=client).rerank("q", HITS, top_n=1)


def test_voyage_reranker_without_a_key_fails_at_construction():
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageReranker(api_key=None)


# --- selection by config -----------------------------------------------------------------------


def settings_from(tmp_path, text):
    path = tmp_path / "hackstack.yaml"
    path.write_text(text, encoding="utf-8")
    return load_settings(path)


def test_build_reranker_noop(tmp_path):
    assert isinstance(
        build_reranker(settings_from(tmp_path, "rerank: {provider: noop}\n")), NoopReranker
    )


def test_build_reranker_cross_encoder_reads_model_and_device(tmp_path):
    settings = settings_from(
        tmp_path,
        "rerank:\n"
        "  {provider: cross_encoder, model: cross-encoder/ms-marco-MiniLM-L6-v2, device: cpu}\n",
    )
    rr = build_reranker(settings, Secrets())
    assert isinstance(rr, CrossEncoderReranker)
    assert (rr.model, rr.device) == ("cross-encoder/ms-marco-MiniLM-L6-v2", "cpu")


def test_build_reranker_voyage_uses_the_key_from_secrets(tmp_path):
    rr = build_reranker(
        settings_from(tmp_path, "rerank: {provider: voyage, model: rerank-3}\n"),
        Secrets(voyage_api_key="vk"),
    )
    assert isinstance(rr, VoyageReranker) and rr.model == "rerank-3"
