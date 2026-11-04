"""Split Markdown into sections by ATX headings, tracking the heading trail."""

from __future__ import annotations

import re

from hackstack.models import Section

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def split_sections(markdown: str) -> list[Section]:
    """One :class:`Section` per heading, plus a leading pathless one for text before the first.

    ``Section.text`` is the body below the heading, without the heading line. A heading
    directly followed by another heading has an empty body. Lines inside fenced code blocks are
    never headings. An empty leading section is omitted.
    """
    sections: list[Section] = []
    trail: list[tuple[int, str]] = []
    path: list[str] = []
    body: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = "\n".join(body).strip("\n")
        if path or text.strip():
            sections.append(Section(path=list(path), text=text))

    for line in markdown.splitlines():
        m_fence = _FENCE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            body.append(line)
            continue
        m = _HEADING.match(line) if fence is None else None
        if not m:
            body.append(line)
            continue
        flush()
        body = []
        level, title = len(m.group(1)), m.group(2).strip()
        trail = [(lvl, t) for lvl, t in trail if lvl < level]
        trail.append((level, title))
        path = [t for _, t in trail]
    flush()
    return sections
