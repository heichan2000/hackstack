"""Live tests against the local models chosen in docs/research/local-models.md.

They skip cleanly when the model is not available, so ``make test`` stays offline:

- ``ollama``: needs Ollama on :11434 with ``qwen3-embedding:0.6b`` pulled
  (``OLLAMA_URL`` overrides the address).
- ``hf_model``: needs the ``rerank`` extra (``uv sync --extra rerank``) and the
  ``cross-encoder/ms-marco-MiniLM-L6-v2`` weights (downloaded on first use, then cached).

Run just these with ``uv run pytest -m "ollama or hf_model"``.
"""

from __future__ import annotations

import math
import os

import httpx
import pytest

from hackstack.embed.ollama import OllamaEmbedder
from hackstack.embed.openai_compat import OpenAICompatEmbedder
from hackstack.models import SearchHit
from hackstack.retrieval.rerank import CrossEncoderReranker

EMBED_MODEL = "qwen3-embedding:0.6b"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
INSTRUCTION = "Given a question, retrieve passages from company documents that answer it"

DOCS = [
    "Unused PTO carries over to the next year, up to a maximum of five days.",
    "The Berlin office cafeteria serves lunch from noon until two.",
    "Deploys on Fridays need a second reviewer and a rollback plan.",
]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def ollama_url():
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        tags = httpx.get(f"{url}/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError) as exc:
        pytest.skip(f"Ollama not reachable at {url}: {exc}")
    names = {m["name"] for m in tags.get("models", [])}
    if EMBED_MODEL not in names and f"{EMBED_MODEL}:latest" not in names:
        pytest.skip(f"{EMBED_MODEL} not pulled (ollama pull {EMBED_MODEL})")
    return url


@pytest.mark.ollama
def test_ollama_embedder_live_returns_1024_unit_vectors_and_ranks_the_answer_first(ollama_url):
    emb = OllamaEmbedder(EMBED_MODEL, 1024, base_url=ollama_url, query_instruction=INSTRUCTION)

    doc_vecs = emb.embed_documents(DOCS)
    query = emb.embed_query("How many vacation days can I roll over?")

    assert len(doc_vecs) == 3 and all(len(v) == 1024 for v in [*doc_vecs, query])
    assert all(math.isclose(math.sqrt(cosine(v, v)), 1.0, abs_tol=1e-3) for v in doc_vecs)
    sims = [cosine(query, d) for d in doc_vecs]
    assert sims.index(max(sims)) == 0, sims


@pytest.mark.ollama
def test_ollama_query_instruction_changes_the_query_vector(ollama_url):
    plain = OllamaEmbedder(EMBED_MODEL, 1024, base_url=ollama_url)
    instructed = OllamaEmbedder(
        EMBED_MODEL, 1024, base_url=ollama_url, query_instruction=INSTRUCTION
    )
    q = "How many vacation days can I roll over?"
    assert cosine(plain.embed_query(q), instructed.embed_query(q)) < 0.999


@pytest.mark.ollama
def test_openai_compat_embedder_live_against_ollama_v1_matches_native(ollama_url):
    native = OllamaEmbedder(EMBED_MODEL, 1024, base_url=ollama_url)
    compat = OpenAICompatEmbedder(EMBED_MODEL, 1024, base_url=f"{ollama_url}/v1")
    (a,) = native.embed_documents([DOCS[0]])
    (b,) = compat.embed_documents([DOCS[0]])
    assert cosine(a, b) > 0.99


@pytest.mark.ollama
def test_ollama_matryoshka_truncation_to_512_live(ollama_url):
    full = OllamaEmbedder(EMBED_MODEL, 1024, base_url=ollama_url)
    small = OllamaEmbedder(EMBED_MODEL, 512, base_url=ollama_url)
    (f,) = full.embed_documents([DOCS[0]])
    (s,) = small.embed_documents([DOCS[0]])
    assert len(s) == 512 and math.isclose(math.sqrt(cosine(s, s)), 1.0, abs_tol=1e-6)
    scale = math.sqrt(cosine(f[:512], f[:512]))
    assert cosine(s, [x / scale for x in f[:512]]) > 0.9999


@pytest.fixture(scope="module")
def cross_encoder():
    pytest.importorskip("sentence_transformers", reason="needs `uv sync --extra rerank`")
    rr = CrossEncoderReranker(RERANK_MODEL, device="cpu")
    try:
        rr.load()
    except OSError as exc:  # weights not cached and no network
        pytest.skip(f"cannot load {RERANK_MODEL}: {exc}")
    return rr


@pytest.mark.hf_model
def test_cross_encoder_live_puts_the_relevant_passage_first(cross_encoder):
    hits = [
        SearchHit(chunk_id=f"c{i}", doc_id=f"d{i}", text=t, score=0.0, retriever="hybrid")
        for i, t in enumerate([DOCS[1], DOCS[2], DOCS[0]])
    ]
    out = cross_encoder.rerank("How many vacation days can I roll over?", hits, top_n=2)
    assert len(out) == 2
    assert out[0].chunk_id == "c2", [(h.chunk_id, h.score) for h in out]
    assert out[0].score > out[1].score
    assert all(h.retriever == "rerank" for h in out)
