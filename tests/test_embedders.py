"""Embedders: Ollama, Voyage and OpenAI-compatible, all against mocked HTTP.

The Voyage responses in ``tests/fixtures/voyage`` follow the shape in Voyage's API reference
(https://docs.voyageai.com/reference/embeddings-api). They were not recorded from the live API:
the kit has no Voyage key yet.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest

from hackstack import registry
from hackstack._http import ProviderError
from hackstack.config import Secrets, load_settings
from hackstack.embed import build_embedder
from hackstack.embed.base import Embedder, fit_dim
from hackstack.embed.ollama import OllamaEmbedder
from hackstack.embed.openai_compat import OpenAICompatEmbedder
from hackstack.embed.voyage import VoyageEmbedder

FIXTURES = Path(__file__).parent / "fixtures" / "voyage"


def norm(vec):
    return math.sqrt(sum(x * x for x in vec))


def unit(n, seed=1.0):
    vec = [math.sin(seed * (i + 1)) for i in range(n)]
    s = norm(vec)
    return [x / s for x in vec]


class Mock:
    """An httpx transport that records requests and answers from a handler."""

    def __init__(self, handler):
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def no_sleep(_seconds):
    pass


# --- fit_dim -----------------------------------------------------------------------------------


def test_fit_dim_keeps_a_vector_of_the_right_length_as_is():
    vec = unit(8)
    assert fit_dim(vec, 8, model="m") == vec


def test_fit_dim_truncates_matryoshka_style_and_renormalizes():
    vec = unit(16)
    fitted = fit_dim(vec, 8, model="m")
    assert len(fitted) == 8
    assert math.isclose(norm(fitted), 1.0, rel_tol=1e-9)
    # same direction as the first 8 dims
    scale = norm(vec[:8])
    assert all(math.isclose(a, b / scale) for a, b in zip(fitted, vec[:8], strict=True))


def test_fit_dim_refuses_a_vector_shorter_than_the_index():
    with pytest.raises(ValueError, match=r"m returned 4 dims but embed\.dim is 8"):
        fit_dim(unit(4), 8, model="m")


# --- Ollama ------------------------------------------------------------------------------------


def ollama_handler(dim=1024):
    def handler(request):
        body = json.loads(request.content)
        inputs = body["input"]
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "embeddings": [unit(dim, i + 1) for i in range(len(inputs))],
            },
        )

    return handler


def test_ollama_embedder_satisfies_the_interface_and_is_registered():
    assert registry.get("embedder", "ollama") is OllamaEmbedder
    assert isinstance(
        OllamaEmbedder("qwen3-embedding:0.6b", 1024, client=Mock(ollama_handler()).client()),
        Embedder,
    )


def test_ollama_embeds_documents_in_batches_via_api_embed():
    mock = Mock(ollama_handler())
    emb = OllamaEmbedder("qwen3-embedding:0.6b", 1024, batch_size=2, client=mock.client())

    vectors = emb.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3 and all(len(v) == 1024 for v in vectors)
    assert [r.url.path for r in mock.requests] == ["/api/embed", "/api/embed"]
    assert [b["input"] for b in mock.bodies()] == [["a", "b"], ["c"]]
    assert all(b["model"] == "qwen3-embedding:0.6b" for b in mock.bodies())


def test_ollama_documents_are_unprefixed_and_queries_get_the_qwen_instruction():
    mock = Mock(ollama_handler())
    emb = OllamaEmbedder(
        "qwen3-embedding:0.6b",
        1024,
        query_instruction="Given a question, retrieve passages that answer it",
        client=mock.client(),
    )

    emb.embed_documents(["PTO carries over up to 5 days."])
    vec = emb.embed_query("How much PTO carries over?")

    assert len(vec) == 1024
    doc_body, query_body = mock.bodies()
    assert doc_body["input"] == ["PTO carries over up to 5 days."]
    assert query_body["input"] == [
        "Instruct: Given a question, retrieve passages that answer it\n"
        "Query: How much PTO carries over?"
    ]


def test_ollama_query_without_an_instruction_is_sent_verbatim():
    mock = Mock(ollama_handler())
    OllamaEmbedder("bge-m3", 1024, client=mock.client()).embed_query("hello")
    assert mock.bodies()[0]["input"] == ["hello"]


def test_ollama_truncates_wider_vectors_to_the_configured_dim():
    mock = Mock(ollama_handler(dim=2048))
    (vec,) = OllamaEmbedder("big", 1024, client=mock.client()).embed_documents(["x"])
    assert len(vec) == 1024 and math.isclose(norm(vec), 1.0, rel_tol=1e-9)


def test_ollama_base_url_accepts_the_openai_style_v1_suffix():
    mock = Mock(ollama_handler())
    emb = OllamaEmbedder("m", 1024, base_url="http://gpu-box:11434/v1/", client=mock.client())
    emb.embed_query("q")
    assert str(mock.requests[0].url) == "http://gpu-box:11434/api/embed"


def test_ollama_missing_model_error_says_how_to_pull_it():
    mock = Mock(lambda r: httpx.Response(404, json={"error": "model 'nope' not found"}))
    emb = OllamaEmbedder("nope", 1024, client=mock.client(), sleep=no_sleep)
    with pytest.raises(ProviderError, match=r"(?s)model 'nope' not found.*ollama pull nope"):
        emb.embed_query("q")


def test_ollama_unreachable_error_says_to_start_ollama():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    emb = OllamaEmbedder("m", 1024, client=Mock(refuse).client(), sleep=no_sleep)
    with pytest.raises(ProviderError, match=r"(?s)cannot reach.*Is Ollama running"):
        emb.embed_documents(["x"])


def test_empty_input_makes_no_request():
    mock = Mock(ollama_handler())
    assert OllamaEmbedder("m", 1024, client=mock.client()).embed_documents([]) == []
    assert mock.requests == []


# --- Voyage (fixtures shaped like the API reference; not recorded live) -------------------------


def voyage_fixture(name):
    return json.loads((FIXTURES / name).read_text())


def voyage_handler(request):
    body = json.loads(request.content)
    name = "embeddings_query.json" if body["input_type"] == "query" else "embeddings_documents.json"
    data = voyage_fixture(name)
    # the API returns data in input order with an index; shuffle to prove we sort by index
    data["data"] = list(reversed(data["data"]))
    return httpx.Response(200, json=data)


def test_voyage_embedder_is_registered_and_satisfies_the_interface():
    assert registry.get("embedder", "voyage") is VoyageEmbedder
    emb = VoyageEmbedder("voyage-4", 256, api_key="k", client=Mock(voyage_handler).client())
    assert isinstance(emb, Embedder)


def test_voyage_documents_use_input_type_document_and_output_dimension():
    mock = Mock(voyage_handler)
    emb = VoyageEmbedder("voyage-4", 256, api_key="sk-test", client=mock.client())

    vectors = emb.embed_documents(["first chunk", "second chunk"])

    expected = voyage_fixture("embeddings_documents.json")["data"]
    assert vectors == [expected[0]["embedding"], expected[1]["embedding"]]
    (request,) = mock.requests
    assert str(request.url) == "https://api.voyageai.com/v1/embeddings"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "model": "voyage-4",
        "input": ["first chunk", "second chunk"],
        "input_type": "document",
        "output_dimension": 256,
    }


def test_voyage_query_uses_input_type_query():
    mock = Mock(voyage_handler)
    vec = VoyageEmbedder("voyage-4", 256, api_key="k", client=mock.client()).embed_query("q")
    assert vec == voyage_fixture("embeddings_query.json")["data"][0]["embedding"]
    assert mock.bodies()[0]["input_type"] == "query"


def test_voyage_retries_a_rate_limit_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"detail": "slow down"})
        return voyage_handler(request)

    slept = []
    emb = VoyageEmbedder(
        "voyage-4", 256, api_key="k", client=Mock(handler).client(), sleep=slept.append
    )
    assert len(emb.embed_query("q")) == 256
    assert len(calls) == 2 and slept == [2.0]


def test_voyage_auth_error_is_not_retried_and_names_the_status():
    mock = Mock(lambda r: httpx.Response(401, json={"detail": "Provided API key is invalid."}))
    emb = VoyageEmbedder("voyage-4", 256, api_key="bad", client=mock.client(), sleep=no_sleep)
    with pytest.raises(ProviderError, match=r"voyage: HTTP 401.*API key is invalid"):
        emb.embed_query("q")
    assert len(mock.requests) == 1


def test_voyage_without_a_key_fails_at_construction_with_a_hint():
    with pytest.raises(ValueError, match=r"VOYAGE_API_KEY.*HACKSTACK_PROFILE=local"):
        VoyageEmbedder("voyage-4", 1024, api_key=None)


# --- OpenAI-compatible -------------------------------------------------------------------------


def openai_handler(dim=1024):
    def handler(request):
        body = json.loads(request.content)
        n = len(body["input"])
        data = [
            {"object": "embedding", "index": i, "embedding": unit(dim, i + 1)} for i in range(n)
        ]
        return httpx.Response(
            200, json={"object": "list", "data": data[::-1], "model": body["model"]}
        )

    return handler


def test_openai_compat_posts_to_base_url_embeddings_and_orders_by_index():
    mock = Mock(openai_handler())
    emb = OpenAICompatEmbedder(
        "text-embedding-3-large",
        1024,
        base_url="https://api.openai.com/v1",
        api_key="sk",
        client=mock.client(),
    )

    vectors = emb.embed_documents(["a", "b"])

    assert vectors[0] == unit(1024, 1) and vectors[1] == unit(1024, 2)
    (request,) = mock.requests
    assert str(request.url) == "https://api.openai.com/v1/embeddings"
    assert request.headers["authorization"] == "Bearer sk"
    assert json.loads(request.content) == {"model": "text-embedding-3-large", "input": ["a", "b"]}


def test_openai_compat_sends_no_auth_header_without_a_key_and_applies_the_instruction():
    mock = Mock(openai_handler())
    emb = OpenAICompatEmbedder(
        "qwen3-embedding:0.6b",
        1024,
        base_url="http://localhost:11434/v1",
        query_instruction="Find it",
        client=mock.client(),
    )
    emb.embed_query("where?")
    assert "authorization" not in mock.requests[0].headers
    assert mock.bodies()[0]["input"] == ["Instruct: Find it\nQuery: where?"]


def test_openai_compat_truncates_and_renormalizes_wide_vectors():
    mock = Mock(openai_handler(dim=3072))
    emb = OpenAICompatEmbedder("m", 1024, base_url="http://x/v1", client=mock.client())
    (vec,) = emb.embed_documents(["a"])
    assert len(vec) == 1024 and math.isclose(norm(vec), 1.0, rel_tol=1e-9)


def test_openai_compat_needs_a_base_url():
    with pytest.raises(ValueError, match="embed.base_url"):
        OpenAICompatEmbedder("m", 1024, base_url=None)


# --- selection by config -----------------------------------------------------------------------


def write_yaml(tmp_path, text):
    path = tmp_path / "hackstack.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_build_embedder_picks_ollama_from_the_local_profile_settings(tmp_path):
    path = write_yaml(
        tmp_path,
        """
embed: {provider: voyage, model: voyage-4, dim: 1024}
profiles:
  local:
    embed:
      provider: ollama
      model: qwen3-embedding:0.6b
      dim: 1024
      base_url: http://localhost:11434
      query_instruction: Answer it
      batch_size: 16
""",
    )
    emb = build_embedder(load_settings(path, profile="local"), Secrets())
    assert isinstance(emb, OllamaEmbedder)
    assert (emb.model, emb.dim, emb.batch_size, emb.query_instruction) == (
        "qwen3-embedding:0.6b",
        1024,
        16,
        "Answer it",
    )


def test_build_embedder_gives_voyage_the_key_from_secrets(tmp_path):
    path = write_yaml(tmp_path, "embed: {provider: voyage, model: voyage-4, dim: 1024}\n")
    emb = build_embedder(load_settings(path), Secrets(voyage_api_key="vk"))
    assert isinstance(emb, VoyageEmbedder) and emb.dim == 1024


def test_build_embedder_constructs_a_component_without_from_settings(tmp_path):
    import tests.fakes  # noqa: F401  registers "fake"

    path = write_yaml(tmp_path, "embed: {provider: fake, model: fake-embed, dim: 16}\n")
    emb = build_embedder(load_settings(path), Secrets())
    assert (emb.model, emb.dim) == ("fake-embed", 16)
    assert len(emb.embed_query("x")) == 16


def test_build_embedder_unknown_provider_lists_the_choices(tmp_path):
    path = write_yaml(tmp_path, "embed: {provider: nope}\n")
    with pytest.raises(KeyError, match=r"ollama.*openai_compat.*voyage"):
        build_embedder(load_settings(path), Secrets())
