"""Strip text a human reader cannot see before it reaches the index.

Hidden text is a prompt-injection channel (OWASP LLM08): white-on-white or ``display:none``
HTML, Word's hidden runs, HTML comments, and invisible Unicode (zero-width characters, bidi
overrides, tag characters used for "ASCII smuggling"). The ingestor runs
:func:`sanitize_content` on raw content before parsing, and :func:`sanitize_text` on the parsed
Markdown after, so whatever a parser lets through is caught too.

Not covered: hidden text inside PDFs (white or off-page text). It needs layout information
that the pypdf fallback does not expose. Docling's layout model is the better place for it.

Known trade-off: the zero-width joiner (U+200D) is removed, which splits composed emoji such as
👩\u200d💻 into their parts. That is harmless for retrieval.
"""

from __future__ import annotations

import io
import re
import zipfile

from bs4 import BeautifulSoup, Comment

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
HTML_MIMES = {"text/html", "application/xhtml+xml"}

# Zero-width and invisible formatting characters, bidi embeddings/overrides/isolates,
# the Mongolian vowel separator, the BOM, and the Unicode tag block (U+E0000-U+E007F).
_INVISIBLE = re.compile(
    "[\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\U000e0000-\U000e007f]"
)
_COMMENT = re.compile(r"<!--.*?-->\n?", re.DOTALL)

# Inline styles that hide an element from a human reader.
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|font-size\s*:\s*0(?:\.0+)?\s*(?:px|pt|em|rem|%)?\s*(?:;|$)"
    r"|opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)"
    r"|(?<![-\w])color\s*:\s*(?:white|#fff(?:fff)?\b|rgba?\(\s*255\s*,\s*255\s*,\s*255)",
    re.IGNORECASE,
)

# A hidden element written inline in Markdown, e.g. <span style="display:none">...</span>.
# Only simple, non-nested elements are matched; real HTML documents go through BeautifulSoup.
_INLINE_HIDDEN = re.compile(
    r"<(?P<tag>[a-zA-Z][a-zA-Z0-9]*)\b(?P<attrs>[^>]*)>.*?</(?P=tag)\s*>", re.DOTALL
)


def sanitize_text(text: str) -> str:
    """Remove invisible Unicode and HTML comments from text or Markdown."""
    return _INVISIBLE.sub("", _COMMENT.sub("", text))


def sanitize_html(html: str) -> str:
    """Remove comments, scripts and hidden elements from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(["script", "style", "template", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.decomposed:
            continue
        if _is_hidden(tag.attrs):
            tag.decompose()
    return sanitize_text(str(soup))


def sanitize_markdown(md: str) -> str:
    """Remove comments, invisible characters and simple inline hidden HTML from Markdown."""

    def drop_hidden(m: re.Match[str]) -> str:
        return "" if _attrs_hide(m.group("attrs")) else m.group(0)

    return sanitize_text(_INLINE_HIDDEN.sub(drop_hidden, md))


def strip_docx_hidden(data: bytes) -> bytes:
    """Return the .docx with every hidden run (``<w:vanish/>``) removed from its XML parts."""
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            payload = src.read(item.filename)
            if item.filename.startswith("word/") and item.filename.endswith(".xml"):
                payload = _strip_vanish_runs(payload.decode("utf-8")).encode("utf-8")
            dst.writestr(item, payload)
    return out.getvalue()


def sanitize_content(content: str | bytes, mime: str) -> str | bytes:
    """Pre-parse sanitizing, routed by MIME. Unknown binary formats pass through unchanged."""
    if mime in HTML_MIMES:
        return sanitize_html(_text(content))
    if mime == DOCX_MIME and isinstance(content, bytes):
        return strip_docx_hidden(content)
    if mime.startswith("text/"):
        return sanitize_markdown(_text(content))
    return content


# --- helpers ---------------------------------------------------------------------------------

_RUN = re.compile(r"<w:r(?:\s[^>]*)?>(?:(?!</w:r>).)*?</w:r>", re.DOTALL)
_VANISH = re.compile(
    r"<w:rPr\b[^>]*>(?:(?!</w:rPr>).)*<w:vanish(?:\s+w:val=\"(?:true|1|on)\")?\s*/>", re.DOTALL
)


def _strip_vanish_runs(xml: str) -> str:
    return _RUN.sub(lambda m: "" if _VANISH.search(m.group(0)) else m.group(0), xml)


def _is_hidden(attrs: dict) -> bool:
    if "hidden" in attrs:
        return True
    style = attrs.get("style") or ""
    return bool(_HIDDEN_STYLE.search(style))


_ATTR_HIDDEN = re.compile(r"(?:^|\s)hidden(?:\s|=|$)", re.IGNORECASE)
_ATTR_STYLE = re.compile(r"style\s*=\s*(\"[^\"]*\"|'[^']*')", re.IGNORECASE)


def _attrs_hide(attrs: str) -> bool:
    if _ATTR_HIDDEN.search(attrs):
        return True
    m = _ATTR_STYLE.search(attrs)
    return bool(m and _HIDDEN_STYLE.search(m.group(1).strip("\"'")))


def _text(content: str | bytes) -> str:
    return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
