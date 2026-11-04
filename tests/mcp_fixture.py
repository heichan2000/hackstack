"""A tiny ACL'd corpus behind the real MCP server, for offline MCP tests.

``build_kb()`` gives a :class:`KBServer` over an in-memory store with the fake embedder, using
the real ``company_brain`` users. Run as a script it serves that over stdio, which is how
``tests/test_mcp_server.py`` drives the stdio transport without Postgres or Ollama::

    python tests/mcp_fixture.py eng@northwind.test
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as a script from the stdio test
    sys.path.insert(0, str(REPO))

from hackstack.config import load_settings  # noqa: E402
from hackstack.mcp_server import KBServer, build_server  # noqa: E402
from hackstack.models import ACL, Chunk, RawDoc  # noqa: E402
from hackstack.retrieval.hybrid import HybridRetriever  # noqa: E402
from hackstack.retrieval.rerank import NoopReranker  # noqa: E402
from tests.fake_store import InMemoryStore  # noqa: E402
from tests.fakes import FakeEmbedder  # noqa: E402

APPS = REPO / "apps"

CORPUS = {
    "handbook/pto.md": (
        "PTO Policy",
        ["group:everyone"],
        [
            "PTO carryover: employees may carry over up to five unused PTO days.",
            "PTO requests go through the HR portal at least two weeks ahead.",
        ],
    ),
    "eng/deploys.md": (
        "Deploy Runbook",
        ["group:engineering"],
        ["Production deploys are frozen on Fridays after 14:00 UTC."],
    ),
    "hr/comp-bands.md": (
        "Compensation Bands",
        ["group:hr"],
        ["Salary band E4 is 165k to 207k USD."],
    ),
}


def build_retriever() -> HybridRetriever:
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


def build_kb(user: str | None = None) -> KBServer:
    settings = load_settings(REPO / "hackstack.yaml", profile="default", env_file=None)
    return KBServer(settings, user=user, apps_dir=APPS, retriever=build_retriever())


if __name__ == "__main__":
    build_server(build_kb(sys.argv[1] if len(sys.argv) > 1 else None)).run("stdio")
