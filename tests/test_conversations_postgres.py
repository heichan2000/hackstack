"""Conversation memory on the Compose Postgres (``make up``); marked ``pg``, skipped without it."""

from __future__ import annotations

import pytest

from hackstack.store.conversations import (
    ConversationConflictError,
    InMemoryConversations,
    PostgresConversations,
)

pytestmark = pytest.mark.pg

TURN = [
    {"role": "user", "content": "PTO carryover?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig=="},
            {"type": "tool_use", "id": "t1", "name": "search_kb", "input": {"query": "PTO"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    {
                        "type": "search_result",
                        "source": "kb://pto#0",
                        "title": "PTO Policy (updated 2026-03-02)",
                        "content": [{"type": "text", "text": "Five days carry over. ✓"}],
                        "citations": {"enabled": True},
                    }
                ],
            }
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "Five days [1]."}]},
]


@pytest.fixture
def pg_conversations(pg_dsn):
    store = PostgresConversations(pg_dsn)
    yield store
    store.close()


def test_roundtrip_is_verbatim_and_ordered(pg_conversations, pg_dsn):
    s = pg_conversations
    assert s.get("c1") is None
    conv = s.create("c1", as_user="eng@northwind.test", app="company_brain")
    assert conv.as_user == "eng@northwind.test"
    assert s.create("c1", as_user="someone@else", app="x").as_user == conv.as_user  # no takeover
    s.append("c1", TURN[:2], start=0)
    s.append("c1", TURN[2:], start=2)

    reopened = PostgresConversations(pg_dsn)  # a new pool: it came from the database
    try:
        assert reopened.messages("c1") == TURN
        assert reopened.get("c1") == conv
    finally:
        reopened.close()


def test_a_stale_writer_gets_a_conflict(pg_conversations):
    s = pg_conversations
    s.create("c2", as_user="eng@northwind.test", app="company_brain")
    s.append("c2", TURN, start=0)
    with pytest.raises(ConversationConflictError):
        s.append("c2", TURN[:1], start=0)
    assert s.messages("c2") == TURN


def test_in_memory_store_has_the_same_contract():
    s = InMemoryConversations()
    s.create("c", as_user="u", app="a")
    s.append("c", TURN, start=0)
    with pytest.raises(ConversationConflictError):
        s.append("c", TURN, start=0)
    got = s.messages("c")
    got[0]["content"] = "mutated"
    assert s.messages("c") == TURN  # callers get copies
