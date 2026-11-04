"""`make ingest-sample`: the active app's sample corpus, ingested end to end.

Offline: the real ``apps/company_brain/sample_corpus`` through the real connector, parser and
chunker, with the fake embedder and the in-memory store. The live run against Postgres and Ollama
is ``HACKSTACK_PROFILE=local make ingest-sample``.
"""

import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

import tests.fake_store  # noqa: F401 - registers store "memory"
from hackstack.chunking.markdown import MarkdownChunker
from hackstack.cli import app
from hackstack.connectors.local_files import LocalFilesConnector
from hackstack.ingest import Ingestor
from hackstack.parsing import router as router_mod
from hackstack.parsing.router import ParserRouter
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder
from tests.test_northwind_corpus import DOCS

REPO = Path(__file__).resolve().parents[1]
APPS = REPO / "apps"
CORPUS = APPS / "company_brain" / "sample_corpus"
COMP = "local_files:northwind/hr/compensation-bands-2026.md"


@pytest.fixture(autouse=True)
def _no_docling(monkeypatch):
    monkeypatch.setattr(router_mod, "docling_available", lambda: False)


def ingest_corpus(store=None):
    conn = LocalFilesConnector()
    conn.configure({"path": str(CORPUS)}, {})
    store = store or InMemoryStore()
    report = Ingestor(
        connector=conn,
        parser=ParserRouter(),
        chunker=MarkdownChunker(),
        embedder=FakeEmbedder(),
        store=store,
    ).run()
    return report, store


def test_the_northwind_corpus_ingests_whole():
    report, store = ingest_corpus()
    assert report.failures == []
    assert report.docs == len(DOCS) == 30
    assert set(store.docs) == {f"local_files:{rel}" for rel in DOCS}
    assert report.chunks == len(store.chunks) > report.docs


@pytest.mark.parametrize("rel", sorted(DOCS))
def test_each_doc_keeps_its_acl_owners_and_date(rel):
    _, store = ingest_corpus()
    meta, doc = DOCS[rel], store.docs[f"local_files:{rel}"]
    assert sorted(doc.acl.principals) == sorted(meta["acl"])
    assert doc.owners == meta.get("owners", [])
    assert isinstance(meta["updated_at"], date)
    assert doc.updated_at is not None and doc.updated_at.date() == meta["updated_at"]


def test_restricted_hr_chunks_carry_only_hr_and_leadership():
    _, store = ingest_corpus()
    chunks = [c for c in store.chunks.values() if c.doc_id == COMP]
    assert chunks
    for c in chunks:
        assert sorted(c.acl_principals) == ["group:hr", "group:leadership"]


def test_re_ingesting_is_idempotent():
    report, store = ingest_corpus()
    first = set(store.chunks)
    again, store = ingest_corpus(store)
    assert (again.docs, again.chunks) == (report.docs, report.chunks)
    assert set(store.chunks) == first


class AnalyzingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.analyzed = 0

    def analyze(self) -> None:
        self.analyzed += 1


def test_a_bulk_ingest_refreshes_store_statistics_once():
    report, store = ingest_corpus(AnalyzingStore())
    assert report.docs == 30
    assert store.analyzed == 1


def _cli(tmp_path, app_name: str):
    cfg = tmp_path / "hackstack.yaml"
    cfg.write_text(
        f"app: {app_name}\nembed: {{provider: fake}}\nstore: {{provider: memory}}\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app, ["ingest", "--file", str(cfg), "--apps-dir", str(APPS), "--json"]
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_cli_ingests_the_active_apps_corpus(tmp_path):
    out = _cli(tmp_path, "company_brain")
    assert (out["connector"], out["docs"], out["failures"]) == ("local_files", 30, [])


def test_switching_app_to_the_template_ingests_its_corpus_instead(tmp_path):
    out = _cli(tmp_path, "_template")
    expected = len(list((APPS / "_template" / "sample_corpus").rglob("[!_]*.md")))
    assert out["docs"] == expected == 1
    assert out["failures"] == []


def test_make_ingest_sample_runs_hackstack_ingest():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    recipe = re.search(r"^ingest-sample:.*\n((?:\t.*\n)+)", makefile, re.M)
    assert recipe, "ingest-sample needs its own target"
    assert "hackstack ingest" in recipe.group(1)
