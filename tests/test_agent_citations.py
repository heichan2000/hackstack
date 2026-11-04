"""The per-conversation citation registry and the search_result blocks it hands out."""

from __future__ import annotations

from datetime import UTC, datetime

from hackstack.agent.citations import (
    NO_RESULTS,
    CitationRegistry,
    display_title,
    is_low_grounding,
    no_results_block,
)
from hackstack.llm.openai_compat_provider import _to_openai
from hackstack.models import Citation, SearchHit


def hit(chunk_id: str, text: str = "Some text.", **kw) -> SearchHit:
    doc_id = chunk_id.rsplit("#", 1)[0]
    return SearchHit(chunk_id=chunk_id, doc_id=doc_id, text=text, score=1.0, **kw)


def test_display_title_is_human_readable_with_section_and_date():
    h = hit(
        "local_files:pto.md#2",
        title="PTO Policy",
        section_path=["PTO Policy", "Carryover"],
        updated_at=datetime(2026, 3, 2, tzinfo=UTC),
    )
    assert display_title(h) == "PTO Policy > Carryover (updated 2026-03-02)"


def test_display_title_falls_back_to_doc_id_without_a_title():
    assert display_title(hit("local_files:notes.md#0")) == "local_files:notes.md"


def test_add_builds_a_citable_search_result_block_split_into_paragraphs():
    reg = CitationRegistry()
    block = reg.add(hit("d#0", "First paragraph.\n\nSecond paragraph.", title="Doc"))
    assert block["type"] == "search_result"
    assert block["source"] == "kb://d#0"
    assert block["title"] == "Doc"
    assert block["citations"] == {"enabled": True}
    assert [p["text"] for p in block["content"]] == ["First paragraph.", "Second paragraph."]


def test_numbers_count_every_block_in_order_and_resolve_fills_title_and_url():
    reg = CitationRegistry()
    reg.add(hit("a#0", title="A", url="https://a"))
    reg.add(hit("b#1", title="B"))
    reg.add(hit("a#0", title="A", url="https://a"))  # shown again, e.g. by get_document
    assert reg.numbers("a#0") == [1, 3]
    assert reg.numbers("b#1") == [2]
    assert reg.seen("b#1") and not reg.seen("c#0")

    resolved = reg.resolve(Citation(doc_id="a", chunk_id="a#0", cited_text="quote"))
    assert (resolved.title, resolved.url, resolved.cited_text) == ("A", "https://a", "quote")
    unknown = Citation(doc_id="z", chunk_id="z#9")
    assert reg.resolve(unknown) == unknown


def test_registry_rebuilt_from_history_matches_the_openai_compat_numbering():
    """[n] markers from the local model must map to the same chunk as the registry's n."""
    reg = CitationRegistry()
    first = [reg.add(hit("a#0", title="A")), reg.add(hit("b#0", title="B"))]
    second = [reg.add(hit("c#0", title="C"))]
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": first}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "x", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": second}],
        },
    ]
    rebuilt = CitationRegistry.from_messages(messages)
    sources = []
    _to_openai(messages, sources)  # the adapter numbers search results the same way
    assert [s.chunk_id for s in sources] == [r.chunk_id for r in rebuilt.refs]
    assert rebuilt.numbers("c#0") == [3]


def test_no_results_block_is_plain_text():
    assert no_results_block() == {"type": "text", "text": NO_RESULTS}
    assert NO_RESULTS == "No results found."


def test_low_grounding_means_zero_citations():
    assert is_low_grounding([])
    assert not is_low_grounding([Citation(doc_id="a", chunk_id="a#0")])
