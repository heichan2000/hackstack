"""Fitting stored history into the model's context: trim old tool results, then drop turns."""

from __future__ import annotations

import copy

from hackstack.agent.citations import CitationRegistry
from hackstack.agent.history import TRIMMED, estimate_tokens, fit_history
from hackstack.models import SearchHit


def block(chunk_id: str, text: str) -> dict:
    hit = SearchHit(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], text=text, score=1.0)
    return CitationRegistry().add(hit)


def turn(n: int, size: int = 2000) -> list[dict]:
    """One agent turn: question, tool call, a search result of ``size`` chars, answer."""
    return [
        {"role": "user", "content": f"question {n}"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"t{n}", "name": "search_kb", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"t{n}",
                    "content": [block(f"doc{n}#0", "x" * size)],
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {n} [{n}]"}]},
    ]


def convo(turns: int, size: int = 2000) -> list[dict]:
    return [m for n in range(1, turns + 1) for m in turn(n, size)]


def result_texts(messages: list[dict]) -> list[str]:
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_result":
                for sr in b["content"]:
                    out.append(sr["content"][0]["text"])
    return out


def test_under_budget_is_unchanged():
    history = convo(2)
    assert fit_history(history, budget_tokens=100_000) == history


def test_trims_old_tool_results_first_and_keeps_the_latest_turn_whole():
    history = convo(3)
    original = copy.deepcopy(history)
    fitted = fit_history(history, budget_tokens=estimate_tokens(history) - 400, keep_turns=1)

    assert history == original  # input untouched: the stored copy stays full
    assert len(fitted) == len(history)  # trimming keeps every message
    texts = result_texts(fitted)
    assert texts[0].startswith(TRIMMED)
    assert texts[-1] == "x" * 2000  # the most recent turn is never trimmed
    assert estimate_tokens(fitted) <= estimate_tokens(history) - 400


def test_trimming_keeps_citation_numbering_but_trimmed_passages_are_not_seen():
    history = convo(3)
    fitted = fit_history(history, budget_tokens=estimate_tokens(history) // 2, keep_turns=1)
    full = CitationRegistry.from_messages(history)
    reg = CitationRegistry.from_messages(fitted)
    assert [r.number for r in reg.refs] == [r.number for r in full.refs] == [1, 2, 3]
    assert [r.chunk_id for r in reg.refs] == ["doc1#0", "doc2#0", "doc3#0"]
    # A trimmed passage must come back if searched again, so it does not count as shown.
    assert not reg.seen("doc1#0")
    assert reg.seen("doc3#0")


def test_drops_oldest_turns_when_trimming_is_not_enough():
    history = convo(4, size=200)
    budget = estimate_tokens(turn(4, size=200)) + 30  # room for about one turn
    fitted = fit_history(history, budget_tokens=budget, keep_turns=1)
    assert fitted[0] == {"role": "user", "content": fitted[0]["content"]}
    assert isinstance(fitted[0]["content"], str)  # starts at a real question
    assert fitted[-4:] == turn(4, size=200)
    assert len(fitted) < len(history)


def test_never_drops_the_kept_turns_even_over_budget():
    history = convo(2, size=5000)
    fitted = fit_history(history, budget_tokens=10, keep_turns=1)
    assert fitted[-4:] == turn(2, size=5000)


def test_single_shot_turns_trim_their_inline_passages():
    history = [
        {"role": "user", "content": [block("a#0", "y" * 3000), {"type": "text", "text": "q1"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": [block("b#0", "z" * 3000), {"type": "text", "text": "q2"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
    ]
    fitted = fit_history(history, budget_tokens=estimate_tokens(history) - 300, keep_turns=1)
    first = fitted[0]["content"]
    assert first[0]["content"][0]["text"].startswith(TRIMMED)
    assert first[1] == {"type": "text", "text": "q1"}  # the question itself stays
    assert fitted[2] == history[2]


def test_long_plain_text_tool_results_are_trimmed_too():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "g", "name": "get_document"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "g", "content": "w" * 4000}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
    ]
    fitted = fit_history(history, budget_tokens=200, keep_turns=1)
    assert fitted[2]["content"][0]["content"].startswith(TRIMMED)
    assert fitted[2]["content"][0]["tool_use_id"] == "g"  # pairing with the tool_use stays valid
