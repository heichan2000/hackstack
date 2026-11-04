"""Route content to a parser by MIME type, and cache results per content hash.

========================================  ==============================================
MIME                                      Parser
========================================  ==============================================
``text/markdown``, ``text/plain``         passthrough (already Markdown)
``text/html``, Office (docx/pptx/xlsx)    MarkItDown
``application/pdf``                       Docling if installed (``uv add docling``), else pypdf
========================================  ==============================================

Docling is optional because it pulls in PyTorch (~1 GB) and downloads layout models on first
use. Without it, PDFs still parse with pypdf, minus tables and heading structure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hackstack.models import ParsedDoc
from hackstack.parsing.base import Parser
from hackstack.parsing.parsers import (
    DOCX_MIME,
    PPTX_MIME,
    XLSX_MIME,
    MarkItDownParser,
    PassthroughParser,
    PypdfParser,
)
from hackstack.registry import PARSER, register

PDF_MIME = "application/pdf"
PARSE_CACHE_DIR = ".data/parse_cache"  # gitignored; `make reset` may clear it

# Bump when routing or any built-in parser changes output, so cached results are not reused.
ROUTER_VERSION = "1"


class UnsupportedMime(ValueError):
    """No parser is routed for this MIME type."""


def docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def _default_docling() -> Parser:
    from hackstack.parsing.docling_parser import DoclingParser

    return DoclingParser()


@register(PARSER, "router")
class ParserRouter:
    """A :class:`~hackstack.parsing.base.Parser` that delegates by MIME type.

    ``docling``: ``None`` uses Docling for PDFs when it is installed, ``False`` never does.
    ``routes``: extra or overriding ``{mime: parser}`` entries.
    ``cache_dir``: also persist parse results as JSON files (the in-memory cache is always on).
    """

    def __init__(
        self,
        *,
        docling: bool | None = None,
        docling_factory: Callable[[], Parser] = _default_docling,
        routes: dict[str, Parser] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        passthrough = PassthroughParser()
        markitdown = MarkItDownParser()
        self._routes: dict[str, Parser] = {
            "text/markdown": passthrough,
            "text/x-markdown": passthrough,
            "text/plain": passthrough,
            "text/html": markitdown,
            "application/xhtml+xml": markitdown,
            DOCX_MIME: markitdown,
            PPTX_MIME: markitdown,
            XLSX_MIME: markitdown,
            **(routes or {}),
        }
        self._use_docling = docling
        self._docling_factory = docling_factory
        self._pdf: Parser | None = None
        self._memo: dict[str, ParsedDoc] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None

    @classmethod
    def from_settings(cls, settings: Any) -> ParserRouter:
        return cls(cache_dir=PARSE_CACHE_DIR)

    def parse(self, content: bytes | str, mime: str) -> ParsedDoc:
        parser = self._parser_for(mime)
        key = self._key(content, mime, parser)
        if key in self._memo:
            return self._memo[key]
        cached = self._read_cache(key)
        if cached is None:
            cached = parser.parse(content, mime)
            self._write_cache(key, cached)
        self._memo[key] = cached
        return cached

    def _parser_for(self, mime: str) -> Parser:
        mime = mime.split(";")[0].strip().lower()
        if mime in self._routes:
            return self._routes[mime]
        if mime == PDF_MIME:
            if self._pdf is None:
                use = docling_available() if self._use_docling is None else self._use_docling
                self._pdf = self._docling_factory() if use else PypdfParser()
            return self._pdf
        raise UnsupportedMime(f"no parser for MIME type {mime!r}")

    @staticmethod
    def _key(content: bytes | str, mime: str, parser: Parser) -> str:
        data = content.encode("utf-8") if isinstance(content, str) else content
        h = hashlib.sha256()
        for part in (ROUTER_VERSION, type(parser).__qualname__, mime):
            h.update(part.encode("utf-8") + b"\0")
        h.update(data)
        return h.hexdigest()

    def _read_cache(self, key: str) -> ParsedDoc | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return ParsedDoc.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            return None  # a corrupt cache entry is re-parsed, never trusted

    def _write_cache(self, key: str, doc: ParsedDoc) -> None:
        if self._cache_dir is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / f"{key}.json").write_text(doc.model_dump_json(), encoding="utf-8")
