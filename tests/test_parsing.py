import pytest

from hackstack.parsing import router as router_mod
from hackstack.parsing.router import ParserRouter, UnsupportedMime
from hackstack.parsing.sections import split_sections
from hackstack.registry import PARSER, get
from tests.filemakers import make_docx, make_pdf

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_split_sections_tracks_heading_trail_and_ignores_code_fences():
    md = (
        "intro line\n\n# Handbook\n\nwelcome\n\n## PTO\n\n### Carryover\n\nfive days\n\n"
        "```python\n# not a heading\nx = 1\n```\n\n## Expenses\n\nreceipts\n"
    )
    sections = split_sections(md)
    assert [s.path for s in sections] == [
        [],
        ["Handbook"],
        ["Handbook", "PTO"],
        ["Handbook", "PTO", "Carryover"],
        ["Handbook", "Expenses"],
    ]
    assert sections[2].text == ""  # heading-only
    assert "# not a heading" in sections[3].text


def test_markdown_and_plain_text_pass_through():
    r = ParserRouter()
    md = "# Title\n\nbody text\n"
    parsed = r.parse(md, "text/markdown")
    assert parsed.markdown == md
    assert parsed.metadata["parser"] == "passthrough"
    assert [s.path for s in parsed.sections] == [["Title"]]
    assert r.parse(b"plain bytes", "text/plain").markdown == "plain bytes"


def test_html_goes_through_markitdown():
    parsed = ParserRouter().parse("<h1>Head</h1><p>para one</p>", "text/html")
    assert parsed.metadata["parser"] == "markitdown"
    assert parsed.markdown.startswith("# Head")
    assert "para one" in parsed.markdown


def test_docx_goes_through_markitdown():
    parsed = ParserRouter().parse(make_docx(["Quarterly plan", "Ship it"]), DOCX)
    assert parsed.metadata["parser"] == "markitdown"
    assert "Quarterly plan" in parsed.markdown and "Ship it" in parsed.markdown


def test_pdf_falls_back_to_pypdf_without_docling(monkeypatch):
    monkeypatch.setattr(router_mod, "docling_available", lambda: False)
    parsed = ParserRouter().parse(
        make_pdf(["Expense policy", "Receipts over $25"]), "application/pdf"
    )
    assert parsed.metadata["parser"] == "pypdf"
    assert parsed.metadata["pages"] == 1
    assert "Receipts over $25" in parsed.markdown


def test_pdf_uses_docling_when_available(monkeypatch):
    calls = []

    class FakeDocling:
        def parse(self, content, mime):
            calls.append(mime)
            from hackstack.models import ParsedDoc

            return ParsedDoc(markdown="# From docling\n", metadata={"parser": "docling"})

    monkeypatch.setattr(router_mod, "docling_available", lambda: True)
    r = ParserRouter(docling_factory=FakeDocling)
    assert r.parse(make_pdf(["x"]), "application/pdf").metadata["parser"] == "docling"
    assert calls == ["application/pdf"]
    # explicitly disabled: pypdf even when docling is importable
    assert (
        ParserRouter(docling=False).parse(make_pdf(["x"]), "application/pdf").metadata["parser"]
        == "pypdf"
    )


def test_parse_results_are_cached_by_content_hash(tmp_path):
    calls = []

    class Spy:
        def parse(self, content, mime):
            calls.append(content)
            from hackstack.models import ParsedDoc

            return ParsedDoc(markdown=f"parsed {content}", metadata={"parser": "spy"})

    r = ParserRouter(routes={"text/x-spy": Spy()}, cache_dir=tmp_path / "cache")
    a1 = r.parse("aaa", "text/x-spy")
    a2 = r.parse("aaa", "text/x-spy")
    r.parse("bbb", "text/x-spy")
    assert a1 == a2 and calls == ["aaa", "bbb"]
    # the disk cache survives a new router instance
    r2 = ParserRouter(routes={"text/x-spy": Spy()}, cache_dir=tmp_path / "cache")
    assert r2.parse("aaa", "text/x-spy").markdown == "parsed aaa"
    assert calls == ["aaa", "bbb"]


def test_unknown_mime_is_refused():
    with pytest.raises(UnsupportedMime):
        ParserRouter().parse(b"\x89PNG", "image/png")


def test_corrupt_pdf_raises(monkeypatch):
    monkeypatch.setattr(router_mod, "docling_available", lambda: False)
    with pytest.raises(Exception):  # noqa: B017 - any parser error; the ingestor records it
        ParserRouter().parse(b"%PDF-1.4 this is not a pdf", "application/pdf")


def test_router_is_registered():
    assert get(PARSER, "router") is ParserRouter
