"""The built-in parsers: passthrough for Markdown, MarkItDown for HTML/Office, pypdf for PDFs."""

from __future__ import annotations

import io

from hackstack.models import ParsedDoc
from hackstack.parsing.sections import split_sections
from hackstack.registry import PARSER, register

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_EXTENSIONS = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    DOCX_MIME: ".docx",
    PPTX_MIME: ".pptx",
    XLSX_MIME: ".xlsx",
}


def _text(content: bytes | str) -> str:
    return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content


def _doc(markdown: str, parser: str, **metadata: object) -> ParsedDoc:
    return ParsedDoc(
        markdown=markdown,
        sections=split_sections(markdown),
        metadata={"parser": parser, **metadata},
    )


@register(PARSER, "passthrough")
class PassthroughParser:
    """Markdown and plain text are already in the target format."""

    def parse(self, content: bytes | str, mime: str) -> ParsedDoc:
        return _doc(_text(content), "passthrough")


@register(PARSER, "markitdown")
class MarkItDownParser:
    """HTML and Office files via Microsoft's MarkItDown (no layout analysis, no GPU).

    The ``docx`` extra is installed. PowerPoint and Excel need ``markitdown[pptx,xlsx]``.
    """

    def __init__(self) -> None:
        self._md = None

    def parse(self, content: bytes | str, mime: str) -> ParsedDoc:
        from markitdown import MarkItDown, StreamInfo

        if self._md is None:
            self._md = MarkItDown(enable_plugins=False)
        data = content.encode("utf-8") if isinstance(content, str) else content
        info = StreamInfo(mimetype=mime, extension=_EXTENSIONS.get(mime), charset="utf-8")
        result = self._md.convert_stream(io.BytesIO(data), stream_info=info)
        meta = {"title": result.title} if result.title else {}
        return _doc(result.markdown, "markitdown", **meta)


@register(PARSER, "pypdf")
class PypdfParser:
    """Plain text per page. The fallback when Docling is not installed: no tables, no headings."""

    def parse(self, content: bytes | str, mime: str) -> ParsedDoc:
        from pypdf import PdfReader

        data = content.encode("latin-1") if isinstance(content, str) else content
        reader = PdfReader(io.BytesIO(data), strict=True)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        if not any(pages):
            raise ValueError("PDF has no extractable text (scanned? install docling for OCR)")
        return _doc("\n\n".join(p for p in pages if p) + "\n", "pypdf", pages=len(pages))
