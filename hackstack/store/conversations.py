"""Conversation memory: ``conversations`` + ``messages(content_json)`` in Postgres.

The ``/chat`` API rebuilds every turn's history from here, never from the client. Messages are
stored exactly as the agent loop produced them (``AgentResult.messages``): assistant turns with
their tool_use blocks and signatures, user turns with their tool_result / search_result blocks.
Replaying them verbatim keeps the provider happy and keeps citation numbers aligned.

A conversation belongs to the user who started it. Its history holds passages only that user
may read, so continuing it as someone else would leak them; callers must refuse that.

:class:`InMemoryConversations` is the same contract in a dict, for tests and storeless runs.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from importlib import resources
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class Conversation:
    id: str
    as_user: str
    app: str


class ConversationConflictError(RuntimeError):
    """Another writer appended to the conversation first (the seq is already taken)."""


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> Conversation | None: ...

    def create(self, conversation_id: str, *, as_user: str, app: str) -> Conversation: ...

    def messages(self, conversation_id: str) -> list[dict[str, Any]]: ...

    def append(
        self, conversation_id: str, messages: list[dict[str, Any]], *, start: int
    ) -> None: ...

    def close(self) -> None: ...


class InMemoryConversations:
    def __init__(self) -> None:
        self._convs: dict[str, Conversation] = {}
        self._msgs: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> Conversation | None:
        return self._convs.get(conversation_id)

    def create(self, conversation_id: str, *, as_user: str, app: str) -> Conversation:
        with self._lock:
            conv = self._convs.setdefault(
                conversation_id, Conversation(conversation_id, as_user, app)
            )
            self._msgs.setdefault(conversation_id, [])
            return conv

    def messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self._msgs.get(conversation_id, []))

    def append(self, conversation_id: str, messages: list[dict[str, Any]], *, start: int) -> None:
        with self._lock:
            stored = self._msgs.setdefault(conversation_id, [])
            if len(stored) != start:
                raise ConversationConflictError(
                    f"conversation {conversation_id!r} has {len(stored)} messages, not {start}"
                )
            stored.extend(copy.deepcopy(messages))

    def close(self) -> None:
        pass


class PostgresConversations:
    """The two tables on the kit's Postgres, behind a small connection pool."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 4) -> None:
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=True, kwargs={"autocommit": True}
        )
        self.pool.wait(timeout=10)
        self.ensure_schema()

    @classmethod
    def from_settings(cls, settings: Any) -> PostgresConversations:
        return cls(settings.store.dsn)

    def ensure_schema(self) -> None:
        sql = resources.files("hackstack.store").joinpath("conversations.sql")
        with self.pool.connection() as conn:
            conn.execute(sql.read_text(encoding="utf-8"))

    def get(self, conversation_id: str) -> Conversation | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT id, as_user, app FROM conversations WHERE id = %s", (conversation_id,)
            ).fetchone()
        return Conversation(*row) if row else None

    def create(self, conversation_id: str, *, as_user: str, app: str) -> Conversation:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, as_user, app) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (conversation_id, as_user, app),
            )
        conv = self.get(conversation_id)
        assert conv is not None
        return conv

    def messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT role, content_json FROM messages WHERE conversation_id = %s ORDER BY seq",
                (conversation_id,),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def append(self, conversation_id: str, messages: list[dict[str, Any]], *, start: int) -> None:
        if not messages:
            return
        rows = [
            (conversation_id, start + i, m["role"], Jsonb(m["content"]))
            for i, m in enumerate(messages)
        ]
        try:
            with self.pool.connection() as conn, conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO messages (conversation_id, seq, role, content_json) "
                        "VALUES (%s, %s, %s, %s)",
                        rows,
                    )
                conn.execute(
                    "UPDATE conversations SET updated_at = now() WHERE id = %s",
                    (conversation_id,),
                )
        except psycopg.errors.UniqueViolation as exc:
            raise ConversationConflictError(
                f"conversation {conversation_id!r} was appended to concurrently"
            ) from exc

    def close(self) -> None:
        self.pool.close()
