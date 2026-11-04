import asyncio
import math

from hackstack import registry
from hackstack.embed.base import Embedder
from hackstack.llm.base import LLMProvider
from hackstack.models import Stop, TextDelta, ToolCall
from tests.fakes import FakeEmbedder, FakeLLM


def collect(provider, messages, system="sys", tools=()):
    async def run():
        return [e async for e in provider.stream(system, messages, list(tools))]

    return asyncio.run(run())


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


# --- embedder ---


def test_fake_embedder_satisfies_the_embedder_interface():
    assert isinstance(FakeEmbedder(), Embedder)


def test_fake_embedder_is_registered_under_fake():
    assert registry.get("embedder", "fake") is FakeEmbedder


def test_fake_embeddings_have_the_configured_dim_and_unit_length():
    (vec,) = FakeEmbedder(dim=32).embed_documents(["carryover policy for PTO"])
    assert len(vec) == 32
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


def test_fake_embeddings_are_deterministic_across_instances():
    text = "who owns the billing service?"
    assert FakeEmbedder().embed_query(text) == FakeEmbedder().embed_query(text)


def test_fake_embeddings_rank_word_overlap_above_unrelated_text():
    emb = FakeEmbedder()
    q = emb.embed_query("PTO carryover policy")
    related, unrelated = emb.embed_documents(
        ["The PTO carryover policy allows five days.", "Kubernetes cluster upgrade runbook."]
    )
    assert cosine(q, related) > cosine(q, unrelated)


def test_fake_embedding_of_empty_text_is_a_zero_vector_of_right_dim():
    assert FakeEmbedder(dim=8).embed_query("") == [0.0] * 8


# --- LLM ---


def test_fake_llm_satisfies_the_provider_interface():
    assert isinstance(FakeLLM(), LLMProvider)


def test_fake_llm_is_registered_under_fake():
    assert registry.get("llm", "fake") is FakeLLM


def test_unscripted_fake_llm_echoes_the_last_user_message_then_stops():
    events = collect(FakeLLM(), [{"role": "user", "content": "hello there"}])
    assert events == [TextDelta(text="echo: hello there"), Stop(reason="end_turn")]


def test_scripted_fake_llm_plays_one_turn_per_call():
    llm = FakeLLM(
        script=[
            [ToolCall(id="t1", name="search_kb", input={"query": "pto"})],
            [TextDelta(text="Five days.")],
        ]
    )
    msgs = [{"role": "user", "content": "PTO?"}]
    first = collect(llm, msgs)
    second = collect(llm, msgs)
    assert first == [
        ToolCall(id="t1", name="search_kb", input={"query": "pto"}),
        Stop(reason="tool_use"),
    ]
    assert second == [TextDelta(text="Five days."), Stop(reason="end_turn")]


def test_fake_llm_records_each_call_for_assertions():
    llm = FakeLLM()
    collect(llm, [{"role": "user", "content": "hi"}], system="be brief", tools=[{"name": "x"}])
    assert llm.calls == [
        {
            "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "x"}],
            "opts": {},
        }
    ]


def test_fake_llm_reads_text_out_of_content_blocks():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "block text"}]}]
    assert collect(FakeLLM(), msgs)[0] == TextDelta(text="echo: block text")
