"""Incremental sync, offline: a local-files corpus, the in-memory store and a counting embedder."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hackstack.chunking.markdown import MarkdownChunker
from hackstack.cli import app
from hackstack.connectors.local_files import LocalFilesConnector
from hackstack.ingest import Ingestor
from hackstack.parsing.router import ParserRouter
from hackstack.registry import STORE, register
from hackstack.sync.manager import SyncManager
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed_documents(texts)


HANDBOOK = (
    "# Handbook\n\n## PTO\n\nTwenty days a year, carried over until March.\n\n"
    "## Expenses\n\nKeep every receipt and file it within thirty days.\n\n"
    "## Travel\n\nBook economy for flights under six hours.\n"
)


def write_corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "_defaults.yaml").write_text("acl: [group:everyone]\n", encoding="utf-8")
    (root / "handbook.md").write_text(HANDBOOK, encoding="utf-8")
    (root / "launch.md").write_text("# Launch\n\nShip on Friday after review.\n", "utf-8")
    hr = root / "hr"
    hr.mkdir(exist_ok=True)
    (hr / "_defaults.yaml").write_text("acl: [group:hr]\n", encoding="utf-8")
    (hr / "salaries.md").write_text("# Salaries\n\nBands are confidential.\n", "utf-8")
    return root


class Clock:
    """Wall-clock time (file mtimes are real) plus an offset ``tick`` moves forward.

    ``last`` is the time the most recent run started, which it saves as its cursor.
    """

    def __init__(self) -> None:
        self.offset = timedelta(0)
        self.last: datetime | None = None

    def __call__(self) -> datetime:
        self.last = datetime.now(UTC) + self.offset
        return self.last

    def tick(self, minutes: int = 5) -> None:
        self.offset += timedelta(minutes=minutes)


def set_mtime(path: Path, when: datetime) -> None:
    os.utime(path, (when.timestamp(), when.timestamp()))


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def env(tmp_path, clock):
    root = write_corpus(tmp_path / "corpus")
    for path in root.rglob("*"):
        set_mtime(path, datetime.now(UTC) - timedelta(days=1))  # everything predates the first sync
    store = InMemoryStore()
    embedder = CountingEmbedder()

    def manager(**cfg) -> SyncManager:
        conn = LocalFilesConnector()
        conn.configure({"path": str(root), **cfg}, {})
        chunker = MarkdownChunker(size=20, max_size=40, min_size=1)
        ing = Ingestor(conn, ParserRouter(), chunker, embedder=embedder, store=store)
        return SyncManager(ing, now=clock)

    return root, store, embedder, manager


def test_rerun_over_unchanged_data_embeds_nothing(env):
    root, store, embedder, manager = env
    first = manager().run()
    assert first.docs == 3 and first.chunks_embedded == len(embedder.embedded) > 0
    chunks_before = dict(store.chunks)

    embedder.embedded.clear()
    second = manager().run(full=True)  # every doc is loaded; the doc hash skips them

    assert embedder.embedded == []
    assert second.unchanged == 3 and second.docs == 0 and second.chunks_embedded == 0
    assert store.chunks.keys() == chunks_before.keys()


def test_an_edited_doc_re_embeds_only_its_changed_chunks(env):
    root, store, embedder, manager = env
    manager().run()
    handbook = [c for c in store.chunks.values() if c.doc_id == "local_files:handbook.md"]
    assert len(handbook) >= 3, "the fixture needs a multi-chunk doc"

    embedder.embedded.clear()
    edited = HANDBOOK.replace("thirty days", "fourteen days")
    (root / "handbook.md").write_text(edited, encoding="utf-8")
    report = manager().run(full=True)

    assert report.docs == 1 and report.unchanged == 2
    assert len(embedder.embedded) == 1 and "fourteen days" in embedder.embedded[0]
    assert report.chunks_embedded == 1 and report.chunks_reused == len(handbook) - 1
    stored = [c for c in store.chunks.values() if c.doc_id == "local_files:handbook.md"]
    assert any("fourteen days" in c.text for c in stored)
    assert all(c.embedding is not None for c in stored)


def test_touching_a_file_without_changing_it_embeds_nothing(env):
    root, store, embedder, manager = env
    manager().run()
    embedder.embedded.clear()

    set_mtime(root / "launch.md", datetime.now(UTC))
    report = manager().run()  # polled because of its new mtime, then skipped by hash

    assert report.mode == "poll" and report.unchanged == 1 and report.docs == 0
    assert embedder.embedded == []


def test_an_acl_change_rewrites_the_doc_without_re_embedding(env):
    root, store, embedder, manager = env
    manager().run()
    embedder.embedded.clear()

    (root / "hr" / "_defaults.yaml").write_text("acl: [group:hr, group:finance]\n", "utf-8")
    report = manager().run()

    assert report.docs == 1 and embedder.embedded == []
    salaries = [c for c in store.chunks.values() if c.doc_id == "local_files:hr/salaries.md"]
    assert salaries and all("group:finance" in c.acl_principals for c in salaries)


def test_a_deleted_source_doc_disappears_from_the_index(env):
    root, store, embedder, manager = env
    manager().run()
    assert "local_files:launch.md" in store.docs

    (root / "launch.md").unlink()
    report = manager().run(full=True)

    assert report.deleted == 1 and report.unchanged == 2
    assert "local_files:launch.md" not in store.docs
    assert not [c for c in store.chunks.values() if c.doc_id == "local_files:launch.md"]


def test_a_doc_that_fails_to_load_or_parse_keeps_its_indexed_version(env):
    root, store, embedder, manager = env
    manager().run()

    (root / "launch.md").write_text("---\nacl: [unterminated\n---\n# Launch\n", "utf-8")
    report = manager().run()

    assert [f.item_id for f in report.failures] == ["local_files:launch.md"]
    assert report.deleted == 0 and "local_files:launch.md" in store.docs


def test_the_index_mirrors_the_connectors_configured_view(env):
    root, store, embedder, manager = env
    manager().run()  # all three docs indexed
    report = manager(max_items=2).run(full=True)
    # list_ids is capped exactly like load_all (sorted paths), so lowering the cap shrinks the
    # index deterministically instead of leaving orphans that no run would ever refresh.
    assert report.deleted == 1
    assert sorted(store.docs) == ["local_files:handbook.md", "local_files:hr/salaries.md"]


# --- cursors and polling ------------------------------------------------------------------------


def test_a_first_run_is_full_and_saves_the_cursor(env, clock):
    root, store, embedder, manager = env
    report = manager().run()
    assert report.mode == "full"
    assert store.get_cursor("local_files") == {"since": clock.last.isoformat()}


def test_the_cursor_persists_and_the_next_run_polls_only_what_changed(env, clock):
    root, store, embedder, manager = env
    manager().run()
    first_cursor = clock.last

    clock.tick()
    (root / "launch.md").write_text("# Launch\n\nShip on Monday instead.\n", "utf-8")
    set_mtime(root / "launch.md", first_cursor + timedelta(minutes=1))
    embedder.embedded.clear()
    report = manager().run()  # a new manager: the cursor came from the store

    assert report.mode == "poll"
    assert report.docs == 1 and report.unchanged == 0  # the other docs were never loaded
    assert len(embedder.embedded) == 1 and "Monday" in embedder.embedded[0]
    assert store.get_cursor("local_files") == {"since": clock.last.isoformat()}


def test_a_poll_run_still_reconciles_deletions(env, clock):
    root, store, embedder, manager = env
    manager().run()
    clock.tick()
    (root / "launch.md").unlink()

    report = manager().run()

    assert report.mode == "poll" and report.deleted == 1
    assert "local_files:launch.md" not in store.docs


def test_a_renamed_file_with_an_old_mtime_is_still_picked_up(env, clock):
    root, store, embedder, manager = env
    manager().run()
    clock.tick()
    (root / "launch.md").rename(root / "launch-plan.md")  # rename keeps the old mtime

    report = manager().run()

    assert "local_files:launch-plan.md" in store.docs
    assert "local_files:launch.md" not in store.docs
    assert report.deleted == 1


def test_full_ignores_the_cursor(env, clock):
    root, store, embedder, manager = env
    manager().run()
    clock.tick()
    report = manager().run(full=True)
    assert report.mode == "full" and report.unchanged == 3


def test_a_failed_run_does_not_advance_the_cursor(env, clock):
    root, store, embedder, manager = env
    manager().run()
    saved = store.get_cursor("local_files")

    class Boom(RuntimeError):
        pass

    def explode(texts):
        raise Boom("embedder down")

    clock.tick()
    (root / "launch.md").write_text("# Launch\n\nChanged.\n", "utf-8")
    set_mtime(root / "launch.md", clock.last + timedelta(minutes=1))
    embedder.embed_documents = explode
    with pytest.raises(Boom):
        manager().run()
    assert store.get_cursor("local_files") == saved


# --- the CLI --------------------------------------------------------------------------------------

_SHARED = InMemoryStore()
register(STORE, "memory-shared")(lambda: _SHARED)  # one store across CLI invocations


def test_cli_ingest_is_incremental_and_full_forces_a_full_read(tmp_path):
    root = write_corpus(tmp_path / "corpus")
    cfg = tmp_path / "hackstack.yaml"
    cfg.write_text(
        "embed: {provider: fake}\nstore: {provider: memory-shared}\n"
        f"connectors:\n  local_files:\n    path: {root.as_posix()}\n",
        encoding="utf-8",
    )
    _SHARED.__init__()  # reset the shared store

    def ingest(*extra):
        result = CliRunner().invoke(app, ["ingest", "--file", str(cfg), "--json", *extra])
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    first = ingest()
    assert first["mode"] == "full" and first["docs"] == 3 and first["chunks_embedded"] > 0
    second = ingest()
    assert second["mode"] == "poll" and second["docs"] == 0 and second["chunks_embedded"] == 0
    third = ingest("--full")
    assert third["mode"] == "full" and third["unchanged"] == 3

    text = CliRunner().invoke(app, ["ingest", "--file", str(cfg)])
    assert text.exit_code == 0 and "unchanged" in text.output
