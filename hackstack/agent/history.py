"""Fit a stored conversation into the model's context window.

The full history stays in Postgres; this only shapes the copy sent to the model. Two steps, in
order, until the estimate fits ``budget_tokens``:

1. **Trim old tool results.** Passage text in turns before the last ``keep_turns`` is replaced
   by a short placeholder. The ``search_result`` blocks themselves stay (same ``source``,
   ``title``), so citation numbering is unchanged, and a trimmed passage no longer counts as
   "already shown": searching again brings it back (:meth:`CitationRegistry.from_messages`).
2. **Drop the oldest turns** whole, from one user question to the next, if trimming was not
   enough. The last ``keep_turns`` turns are never dropped.

A *turn* starts at a user message that is not just tool results (a typed question, or a
single-shot question with its passages inline). The research doc's advice for long sessions:
"trim old tool-result payloads" (*Memory, caching, MCP, tracing and evals*).
"""

from __future__ import annotations

import copy
import json
from typing import Any

#: Prefix of every trimmed payload; :class:`CitationRegistry` treats such passages as unseen.
TRIMMED = "[trimmed from history"
TRIMMED_TEXT = f"{TRIMMED}: search again or call get_document to re-read it]"
_KEEP_PLAIN = 300  # plain-text tool results up to this many chars are left alone


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """A rough token count (~4 chars per token over the JSON), good enough for budgeting."""
    return len(json.dumps(messages, ensure_ascii=False)) // 4


def _is_question(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    return any(b.get("type") != "tool_result" for b in content or [])


def turn_starts(messages: list[dict[str, Any]]) -> list[int]:
    return [i for i, m in enumerate(messages) if _is_question(m)]


def _trim_search_result(block: dict[str, Any]) -> bool:
    content = block.get("content") or []
    if content and str(content[0].get("text", "")).startswith(TRIMMED):
        return False
    block["content"] = [{"type": "text", "text": TRIMMED_TEXT}]
    return True


def _trim_blocks(blocks: list[dict[str, Any]]) -> None:
    for b in blocks:
        kind = b.get("type")
        if kind == "search_result":
            _trim_search_result(b)
        elif kind == "tool_result":
            inner = b.get("content")
            if isinstance(inner, str):
                if len(inner) > _KEEP_PLAIN and not inner.startswith(TRIMMED):
                    b["content"] = TRIMMED_TEXT
            elif isinstance(inner, list):
                for x in inner:
                    if x.get("type") == "search_result":
                        _trim_search_result(x)
                    elif x.get("type") == "text" and len(x.get("text", "")) > _KEEP_PLAIN:
                        x["text"] = TRIMMED_TEXT


def _trim_message(message: dict[str, Any]) -> None:
    if message.get("role") == "user" and isinstance(message.get("content"), list):
        _trim_blocks(message["content"])


def fit_history(
    messages: list[dict[str, Any]], *, budget_tokens: int, keep_turns: int = 1
) -> list[dict[str, Any]]:
    """``messages`` shrunk to about ``budget_tokens``; the input is never modified."""
    if estimate_tokens(messages) <= budget_tokens:
        return messages
    out = copy.deepcopy(messages)
    starts = turn_starts(out)
    protected = starts[-keep_turns] if keep_turns > 0 and len(starts) >= keep_turns else len(out)
    if not starts:
        protected = len(out)

    # 1. Trim, oldest turn first, re-checking after each turn.
    bounds = [*starts, len(out)]
    first = bounds[0] if bounds else 0
    segments = [(0, first)] if first else []
    segments += list(zip(bounds[:-1], bounds[1:], strict=True))
    for lo, hi in segments:
        if lo >= protected:
            break
        for m in out[lo:hi]:
            _trim_message(m)
        if estimate_tokens(out) <= budget_tokens:
            return out

    # 2. Drop whole turns from the front; the result always starts at a question.
    while estimate_tokens(out) > budget_tokens:
        starts = turn_starts(out)
        if len(starts) <= max(keep_turns, 1):
            break
        out = out[starts[1] :]
    if out and not _is_question(out[0]):
        starts = turn_starts(out)
        out = out[starts[0] :] if starts else out
    return out
