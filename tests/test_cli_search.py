"""``hackstack search``: hybrid retrieval from the command line, as a named user."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hackstack.cli import app
from hackstack.models import ACL, Chunk, RawDoc
from hackstack.retrieval import hybrid
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder

APPS = Path(__file__).resolve().parents[1] / "apps"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A config on the fakes, with the CLI's retriever pointed at a pre-filled memory store."""
    store = InMemoryStore()
    emb = FakeEmbedder()
    for doc_id, acl, text in [
        ("pto", ["group:everyone"], "PTO carryover is five days."),
        ("bands", ["group:hr"], "Compensation bands by level."),
    ]:
        raw = RawDoc(
            doc_id=doc_id,
            source="local_files",
            title=doc_id.upper(),
            content="",
            acl=ACL(principals=acl),
        )
        c = Chunk.for_doc(raw, ordinal=0, text=text)
        c.embedding = emb.embed_query(text)
        store.upsert(raw, [c])

    real = hybrid.HybridRetriever.from_settings.__func__

    def with_store(cls, settings, **kw):
        return real(cls, settings, store=store, **kw)

    monkeypatch.setattr(hybrid.HybridRetriever, "from_settings", classmethod(with_store))
    path = tmp_path / "hackstack.yaml"
    path.write_text(
        "embed: {provider: fake}\nstore: {provider: memory}\nrerank: {provider: noop}\n",
        encoding="utf-8",
    )
    return path


def run(*args):
    return CliRunner().invoke(app, ["search", *args, "--apps-dir", str(APPS)])


def test_search_json_lists_stages_and_hits_for_the_user(cfg):
    result = run("compensation bands", "--as", "hr@northwind.test", "--file", str(cfg), "--json")
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["user"] == "hr@northwind.test"
    assert "group:hr" in out["principals"] and "public" in out["principals"]
    assert out["hits"][0]["doc_id"] == "bands"
    assert set(out["counts"]) == {"dense", "lexical", "fused", "reranked", "hits"}


def test_search_as_another_user_does_not_see_hr_docs(cfg):
    result = run("compensation bands", "--as", "eng@northwind.test", "--file", str(cfg), "--json")
    assert result.exit_code == 0, result.output
    assert "bands" not in {h["doc_id"] for h in json.loads(result.stdout)["hits"]}


def test_search_defaults_to_the_apps_default_user_and_prints_text(cfg):
    result = run("PTO carryover", "--file", str(cfg))
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("eng@northwind.test:")
    assert "PTO" in result.stdout


def test_search_unknown_user_is_a_clean_error(cfg):
    result = run("PTO", "--as", "nobody@northwind.test", "--file", str(cfg))
    assert result.exit_code == 2
    assert "nobody@northwind.test" in result.output
