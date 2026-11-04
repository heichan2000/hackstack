"""The agent loop against the real local model, over a tiny in-memory fixture corpus.

Skipped unless Ollama is up with ``hackstack-chat`` (``make ollama-models``). Run just these:
``uv run pytest -m ollama tests/test_agent_live.py``.
"""

from __future__ import annotations

import asyncio
import os

import httpx2
import pytest

from hackstack.agent.loop import collect, stream_agent
from hackstack.agent.runtime import default_system_prompt
from hackstack.agent.tools import ToolContext, load_tools
from hackstack.llm.openai_compat_provider import OpenAICompatProvider
from hackstack.models import ACL, Chunk, RawDoc
from hackstack.retrieval.hybrid import HybridRetriever
from hackstack.retrieval.rerank import NoopReranker
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.ollama

MODEL = "hackstack-chat"
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ENG = ["user:eng@northwind.test", "group:everyone", "group:engineering", "public"]

CORPUS = {
    "pto-policy": (
        "PTO Policy",
        ["group:everyone"],
        [
            "PTO carryover: employees may carry over up to five unused PTO days into the next "
            "calendar year. Days above five are forfeited on January 1.",
            "PTO requests go through the HR portal at least two weeks ahead.",
        ],
    ),
    "deploy-runbook": (
        "Deploy Runbook",
        ["group:engineering"],
        [
            "Production deploys are frozen on Fridays after 14:00 UTC.",
        ],
    ),
    "comp-bands": (
        "Compensation Bands",
        ["group:hr"],
        [
            "Salary band L5 is 140k to 175k USD.",
        ],
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _ollama_with_model():
    try:
        tags = httpx2.get(f"{OLLAMA}/api/tags", timeout=2).json()
    except httpx2.HTTPError as exc:
        pytest.skip(f"Ollama not reachable at {OLLAMA}: {exc}")
    if not any(m["name"].split(":")[0] == MODEL for m in tags.get("models", [])):
        pytest.skip(f"Ollama has no {MODEL} model; run `make ollama-models`")


@pytest.fixture(scope="module")
def retriever():
    store, emb = InMemoryStore(), FakeEmbedder()
    for doc_id, (title, acl, texts) in CORPUS.items():
        raw = RawDoc(
            doc_id=doc_id, source="local_files", title=title, content="", acl=ACL(principals=acl)
        )
        chunks = [Chunk.for_doc(raw, ordinal=i, text=t) for i, t in enumerate(texts)]
        for c in chunks:
            c.embedding = emb.embed_query(c.text)
        store.upsert(raw, chunks)
    return HybridRetriever(store, emb, NoopReranker(), window=10, top_n=3)


def ask(question, retriever):
    llm = OpenAICompatProvider(model=MODEL, base_url=f"{OLLAMA}/v1", reasoning_effort="none")
    return asyncio.run(
        collect(
            stream_agent(
                llm,
                system=default_system_prompt(app_title="Northwind Brain", user="Alex Chen"),
                messages=[{"role": "user", "content": question}],
                tools=load_tools(["search_kb", "get_document"]),
                ctx=ToolContext(principals=ENG, retriever=retriever),
                max_iterations=4,
                llm_opts={"temperature": 0, "seed": 7},
            )
        )
    )


def test_local_model_answers_with_a_citation(retriever):
    result = ask("How many unused PTO days can I carry over?", retriever)
    assert result.stop_reason == "end_turn", result
    assert any(t.name == "search_kb" for t in result.tool_calls)
    assert "five" in result.answer.lower() or "5" in result.answer
    assert "pto-policy" in {c.doc_id for c in result.citations}
    assert not result.low_grounding


def test_local_model_never_sees_restricted_documents(retriever):
    result = ask("What is the salary range for band L5?", retriever)
    assert "comp-bands" not in {s.doc_id for s in result.sources}
    assert "140" not in result.answer
