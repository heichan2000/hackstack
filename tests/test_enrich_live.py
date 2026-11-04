"""Contextual enrichment against the real local model (``make ollama-models``).

Run just these: ``uv run pytest -m ollama tests/test_enrich_live.py``.
"""

from __future__ import annotations

import os

import httpx2
import pytest

from hackstack.chunking.markdown import MarkdownChunker
from hackstack.enrich.contextual import ContextualEnricher
from hackstack.llm.openai_compat_provider import OpenAICompatProvider
from hackstack.models import ACL, ParsedDoc, RawDoc

pytestmark = pytest.mark.ollama

MODEL = "hackstack-chat"
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

DOC = """# Engineering handbook

Northwind Labs builds warehouse robots. This handbook is owned by Priya Natarajan (eng).

## Code review

Every change to the fleet controller needs one approving review from the controls team.

## Deploys

Production deploys are frozen on Fridays after 14:00 UTC. Hotfixes need sign-off from the
on-call lead.
"""


@pytest.fixture(scope="module", autouse=True)
def _ollama_with_model():
    try:
        tags = httpx2.get(f"{OLLAMA}/api/tags", timeout=2).json()
    except httpx2.HTTPError as exc:
        pytest.skip(f"Ollama not reachable at {OLLAMA}: {exc}")
    if not any(m["name"].split(":")[0] == MODEL for m in tags.get("models", [])):
        pytest.skip(f"Ollama has no {MODEL} model; run `make ollama-models`")


def test_local_model_situates_each_chunk():
    raw = RawDoc(
        doc_id="local_files:eng.md",
        source="local_files",
        title="Engineering handbook",
        content=DOC,
        acl=ACL(principals=["group:eng"]),
    )
    parsed = ParsedDoc(markdown=DOC)
    chunks = MarkdownChunker(size=40, max_size=60, min_size=1).chunk(parsed, raw)
    llm = OpenAICompatProvider(model=MODEL, base_url=f"{OLLAMA}/v1", reasoning_effort="none")

    stats = ContextualEnricher(llm, max_doc_tokens=8000).enrich(raw, parsed, chunks)

    assert stats.enriched == len(chunks) >= 2, stats
    deploys = next(c for c in chunks if "Fridays" in c.text)
    assert deploys.context and len(deploys.context.split()) <= 150
    assert "<think>" not in deploys.context
    assert deploys.context in deploys.embed_text and deploys.embed_text.endswith(deploys.text)
    assert stats.usage.get("input_tokens", 0) > 0
