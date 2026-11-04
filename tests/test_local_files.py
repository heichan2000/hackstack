import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hackstack.connectors.base import ConnectorFailure
from hackstack.connectors.local_files import LocalFilesConnector
from hackstack.models import RawDoc
from hackstack.registry import CONNECTOR, get
from tests.filemakers import make_docx, make_pdf


def write(root: Path, rel: str, content: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def load(root: Path, **cfg):
    c = LocalFilesConnector()
    c.configure({"path": str(root), **cfg}, {})
    c.validate()
    items = list(c.load_all())
    docs = {i.doc_id: i for i in items if isinstance(i, RawDoc)}
    failures = [i for i in items if isinstance(i, ConnectorFailure)]
    return docs, failures


def test_supported_files_become_raw_docs(tmp_path):
    write(tmp_path, "a.md", "# Alpha\n\ntext")
    write(tmp_path, "notes.txt", "plain")
    write(tmp_path, "sub/page.html", "<h1>Page</h1>")
    write(tmp_path, "sub/report.pdf", make_pdf(["pdf text"]))
    write(tmp_path, "sub/plan.docx", make_docx(["docx text"]))
    write(tmp_path, "image.png", b"\x89PNG")
    write(tmp_path, ".git/config", "[core]")
    write(tmp_path, "a.md.meta.yaml", "acl: [public]\n")
    write(tmp_path, "sub/_defaults.yaml", "acl: [group:eng]\n")
    write(tmp_path, "_drafts/wip.md", "not a document")  # `_` marks config, not content
    write(tmp_path, "_notes.md", "not a document either")
    docs, failures = load(tmp_path)
    assert failures == []
    assert sorted(docs) == [
        "local_files:a.md",
        "local_files:notes.txt",
        "local_files:sub/page.html",
        "local_files:sub/plan.docx",
        "local_files:sub/report.pdf",
    ]
    assert docs["local_files:a.md"].mime == "text/markdown"
    assert isinstance(docs["local_files:a.md"].content, str)
    assert docs["local_files:sub/report.pdf"].mime == "application/pdf"
    assert isinstance(docs["local_files:sub/report.pdf"].content, bytes)
    assert docs["local_files:sub/plan.docx"].mime.endswith("wordprocessingml.document")
    assert docs["local_files:sub/page.html"].mime == "text/html"
    assert all(d.source == "local_files" for d in docs.values())


def test_acl_precedence(tmp_path):
    write(tmp_path, "_defaults.yaml", "acl: [group:everyone]\n")
    write(tmp_path, "hr/_defaults.yaml", "acl: [group:hr]\n")
    write(tmp_path, "hr/salaries.md", "pay")
    write(tmp_path, "hr/handbook.md", "rules")
    write(tmp_path, "hr/handbook.md.meta.yaml", "acl: [group:everyone, group:hr]\n")
    write(tmp_path, "hr/review.md", "---\nacl: [user:dana@northwind.test]\n---\nreview")
    write(tmp_path, "general.md", "hello")
    docs, _ = load(tmp_path)
    acl = {k.removeprefix("local_files:"): v.acl.principals for k, v in docs.items()}
    assert acl["general.md"] == ["group:everyone"]  # root folder defaults
    assert acl["hr/salaries.md"] == ["group:hr"]  # nearest folder defaults win
    assert acl["hr/handbook.md"] == ["group:everyone", "group:hr"]  # sidecar beats folder
    assert acl["hr/review.md"] == ["user:dana@northwind.test"]  # front matter beats all


def test_folder_defaults_cascade_key_by_key(tmp_path):
    write(
        tmp_path,
        "_defaults.yaml",
        "acl: [group:everyone]\nsource_type: page\nupdated_at: 2026-03-02\n",
    )
    write(tmp_path, "eng/_defaults.yaml", "owners: [sam@northwind.test]\n")
    write(tmp_path, "eng/runbook.md", "---\nupdated_at: 2026-11-02\n---\nsteps")
    write(tmp_path, "eng/oncall.md", "rota")
    docs, _ = load(tmp_path)
    runbook, oncall = docs["local_files:eng/runbook.md"], docs["local_files:eng/oncall.md"]
    for doc in (runbook, oncall):
        assert doc.acl.principals == ["group:everyone"]  # from the root
        assert doc.owners == ["sam@northwind.test"]  # from eng/
        assert doc.source_type == "page"
    assert runbook.updated_at == datetime(2026, 11, 2, tzinfo=UTC)  # front matter wins
    assert oncall.updated_at == datetime(2026, 3, 2, tzinfo=UTC)


def test_a_bad_folder_defaults_file_fails_every_doc_below_it(tmp_path):
    write(tmp_path, "ok.md", "fine")
    write(tmp_path, "hr/_defaults.yaml", "acl: [group:hr\n")
    write(tmp_path, "hr/a.md", "a")
    write(tmp_path, "hr/b.md", "b")
    docs, failures = load(tmp_path)
    assert list(docs) == ["local_files:ok.md"]  # never emitted with a missing ACL
    assert sorted(f.item_id for f in failures) == ["local_files:hr/a.md", "local_files:hr/b.md"]


def test_without_any_rule_docs_are_private_to_the_ingesting_user(tmp_path):
    write(tmp_path, "x.md", "x")
    docs, _ = load(tmp_path, owner="sam@northwind.test")
    assert docs["local_files:x.md"].acl.principals == ["user:sam@northwind.test"]
    docs, _ = load(tmp_path, default_acl=["group:everyone"])
    assert docs["local_files:x.md"].acl.principals == ["group:everyone"]


def test_front_matter_supplies_metadata_and_is_stripped(tmp_path):
    write(
        tmp_path,
        "policy.md",
        "---\ntitle: Travel policy\nowners: [priya@northwind.test]\n"
        "updated_at: 2026-10-30\nurl: https://wiki.test/travel\nsource_type: page\n"
        "team: finance\n---\n# Travel\n\nBook economy.\n",
    )
    doc = load(tmp_path)[0]["local_files:policy.md"]
    assert doc.title == "Travel policy"
    assert doc.owners == ["priya@northwind.test"]
    assert doc.updated_at == datetime(2026, 10, 30, tzinfo=UTC)
    assert doc.url == "https://wiki.test/travel"
    assert doc.source_type == "page"
    assert doc.extra["team"] == "finance"
    assert doc.content == "# Travel\n\nBook economy.\n"


def test_title_falls_back_to_first_heading_then_file_name(tmp_path):
    write(tmp_path, "with-h1.md", "intro\n\n# Real Title\n\nbody")
    write(tmp_path, "with-h2.md", "## Second level\n\nbody")
    write(tmp_path, "expense_policy-v2.txt", "body")
    docs, _ = load(tmp_path)
    assert docs["local_files:with-h1.md"].title == "Real Title"
    assert docs["local_files:with-h2.md"].title == "Second level"
    assert docs["local_files:expense_policy-v2.txt"].title == "expense policy v2"


def test_updated_at_never_comes_from_the_file_mtime(tmp_path):
    p = write(tmp_path, "x.md", "x")
    ts = datetime(2026, 11, 1, 9, 30, tzinfo=UTC).timestamp()
    os.utime(p, (ts, ts))
    doc = load(tmp_path)[0]["local_files:x.md"]
    assert doc.updated_at is None  # a checkout resets mtimes; "what changed" must not trust them
    assert doc.extra["mtime"] == "2026-11-01T09:30:00+00:00"
    assert doc.extra["path"] == "x.md"


def test_a_bad_sidecar_yields_a_failure_and_the_run_continues(tmp_path):
    write(tmp_path, "good.md", "fine")
    write(tmp_path, "bad.md", "content")
    write(tmp_path, "bad.md.meta.yaml", "acl: [unclosed\n")
    docs, failures = load(tmp_path)
    assert list(docs) == ["local_files:good.md"]
    assert len(failures) == 1
    f = failures[0]
    assert f.connector == "local_files" and f.item_id == "local_files:bad.md"
    assert f.stage == "load" and f.error


def test_max_items_caps_the_backfill(tmp_path):
    for i in range(5):
        write(tmp_path, f"d{i}.md", "x")
    docs, _ = load(tmp_path, max_items=2)
    assert len(docs) == 2


def test_validate_fails_fast_on_a_missing_folder(tmp_path):
    c = LocalFilesConnector()
    c.configure({"path": str(tmp_path / "nope")}, {})
    with pytest.raises(ValueError, match="nope"):
        c.validate()


def test_connector_is_registered():
    assert get(CONNECTOR, "local_files") is LocalFilesConnector
