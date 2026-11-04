"""PDFs via Docling: layout detection, table structure and reading order. Optional.

Install with ``uv add docling`` (pulls in PyTorch). The first run downloads layout models, so
warm it up before a demo. Not exercised by ``make test``: the router falls back to pypdf when
Docling is not importable.
"""

from __future__ import annotations

import io

from hackstack.models import ParsedDoc
from hackstack.parsing.sections import split_sections
from hackstack.registry import PARSER, register


@register(PARSER, "docling")
class DoclingParser:
    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()

    def parse(self, content: bytes | str, mime: str) -> ParsedDoc:
        from docling.datamodel.base_models import DocumentStream

        data = content.encode("latin-1") if isinstance(content, str) else content
        result = self._converter.convert(DocumentStream(name="doc.pdf", stream=io.BytesIO(data)))
        markdown = result.document.export_to_markdown()
        return ParsedDoc(
            markdown=markdown,
            sections=split_sections(markdown),
            metadata={"parser": "docling", "pages": len(result.document.pages)},
        )
