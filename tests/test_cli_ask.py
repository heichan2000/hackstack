"""``hackstack ask``: the agent from the command line, on the fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hackstack.llm
from hackstack.cli import app
from hackstack.models import ACL, Chunk, Citation, CitationEvent, RawDoc, TextDelta, ToolCall
from hackstack.retrieval import hybrid
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder, FakeLLM

APPS = Path(__file__).resolve().parents[1] / "apps"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        hybrid.HybridRetriever,
        "from_settings",
        classmethod(lambda cls, settings, **kw: real(cls, settings, store=store, **kw)),
    )
    path = tmp_path / "hackstack.yaml"
    path.write_text(
        "embed: {provider: fake}\nstore: {provider: memory}\nrerank: {provider: noop}\n",
        encoding="utf-8",
    )
    return path


def scripted(monkeypatch, script):
    llm = FakeLLM(script)
    monkeypatch.setattr(hackstack.llm, "make_llm", lambda settings, *a, **kw: llm)
    return llm


def run(*args):
    return CliRunner().invoke(app, ["ask", *args, "--apps-dir", str(APPS)])


def test_ask_answers_with_citations(cfg, monkeypatch):
    llm = scripted(
        monkeypatch,
        [
            [ToolCall(id="a", name="search_kb", input={"query": "PTO carryover"})],
            [
                TextDelta(text="Five days [1]."),
                CitationEvent(citation=Citation(doc_id="pto", chunk_id="pto#0")),
            ],
        ],
    )
    result = run("What is PTO carryover?", "--as", "eng@northwind.test", "--file", str(cfg))
    assert result.exit_code == 0, result.output
    assert "Five days [1]." in result.stdout
    assert "[1] PTO  (pto#0)" in result.stdout
    assert "Alex Chen" in llm.calls[0]["system"]  # the app's prompt, rendered for the user


def test_ask_json_and_acl(cfg, monkeypatch):
    scripted(
        monkeypatch,
        [
            [ToolCall(id="a", name="search_kb", input={"query": "compensation bands"})],
            [TextDelta(text="I don't know.")],
        ],
    )
    result = run("bands?", "--as", "eng@northwind.test", "--file", str(cfg), "--json")
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["user"] == "eng@northwind.test"
    assert "bands" not in {s["doc_id"] for s in out["sources"]}  # eng cannot see HR docs
    assert out["low_grounding"] is True


def test_ask_single_shot(cfg, monkeypatch):
    llm = scripted(monkeypatch, [[TextDelta(text="Five days.")]])
    result = run("PTO carryover", "--file", str(cfg), "--mode", "single_shot", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["mode"] == "single_shot"
    assert len(llm.calls) == 1 and llm.calls[0]["tools"] == []


def test_ask_unknown_user_is_a_clean_error(cfg, monkeypatch):
    scripted(monkeypatch, [])
    result = run("q", "--as", "nobody@x.test", "--file", str(cfg))
    assert result.exit_code == 2
    assert "nobody@x.test" in result.output


def test_ask_bad_mode(cfg):
    result = run("q", "--file", str(cfg), "--mode", "fast")
    assert result.exit_code == 2
