"""Postgres + pgvector store: HNSW dense search and ``tsvector`` lexical search, both ACL-filtered.

The permission filter (``acl_principals && :principals``) and the ``source`` / ``since`` filters
are built once by :func:`filter_clause` and spliced into *both* queries, so the two retrievers can
never disagree about what a user may see. Fail closed: a document whose ACL is empty is private to
the ingesting user, or visible to nobody when there is no ingesting user, and a query with no
principals matches nothing.

The index remembers which embedder built it (``collection_meta``). Opening it with a different
``embed.model`` or ``embed.dim``, or writing or querying vectors of the wrong length, raises
:class:`IndexMismatchError`: run ``make reset`` and re-ingest.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from importlib import resources
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from hackstack.models import Chunk, RawDoc, SearchHit
from hackstack.registry import STORE, register

if TYPE_CHECKING:
    from hackstack.config import Settings

#: Lower bound for ``hnsw.ef_search``; it is raised to ``k`` when more candidates are asked for.
#: pgvector's default (40) silently caps how many rows the index returns.
EF_SEARCH_MIN = 100
EF_SEARCH_MAX = 1000  # pgvector's upper limit

FILTER_KEYS = frozenset({"source", "since"})

_ADVISORY_LOCK = 7_461_726  # serializes concurrent ensure_schema/reset calls ("hackstack")

_TABLES = ("chunks", "documents", "sync_state", "collection_meta")


class IndexMismatchError(RuntimeError):
    """The index was built by a different embedder (model or dimension) than the one in use."""


# --- pure helpers (unit-tested offline) -------------------------------------------------------


def render_schema(dim: int) -> str:
    """``schema.sql`` with the vector dimension filled in."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ValueError(f"embed.dim must be a positive integer, got {dim!r}")
    sql = resources.files("hackstack.store").joinpath("schema.sql").read_text(encoding="utf-8")
    return sql.replace("__DIM__", str(dim))


def vector_literal(vec: Iterable[float]) -> str:
    """pgvector's text form, e.g. ``[1.0,0.5]``."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def filter_clause(
    principals: Sequence[str], filters: dict[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    """The WHERE clause shared by both retrievers (over chunks aliased ``c``) and its params.

    Supported filters: ``source`` (a name or list of names) and ``since`` (datetime, date or ISO
    string; naive values are UTC). ``None`` values are ignored; unknown keys raise.
    """
    filters = {k: v for k, v in (filters or {}).items() if v is not None}
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise ValueError(
            f"unknown search filter(s) {sorted(unknown)}; supported: {sorted(FILTER_KEYS)}"
        )

    clauses = ["c.acl_principals && %(principals)s::text[]"]
    params: dict[str, Any] = {"principals": list(principals)}
    if "source" in filters:
        src = filters["source"]
        params["sources"] = [src] if isinstance(src, str) else list(src)
        clauses.append("c.source = ANY(%(sources)s)")
    if "since" in filters:
        params["since"] = _as_utc(filters["since"])
        clauses.append("c.updated_at >= %(since)s")
    return " AND ".join(clauses), params


def effective_principals(doc: RawDoc, *, ingest_user: str | None) -> list[str]:
    """The document's ACL, or private to the ingesting user when it has none (fail closed)."""
    if doc.acl.principals:
        return list(doc.acl.principals)
    return [f"user:{ingest_user}"] if ingest_user else []


def _as_utc(value: datetime | date | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):  # a plain date: midnight UTC
        value = datetime(value.year, value.month, value.day)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _content_hash(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


# --- SQL --------------------------------------------------------------------------------------

_HIT_COLUMNS = """
    c.chunk_id, c.doc_id, c.text, c.section_path, c.source, c.updated_at,
    coalesce(d.title, '') AS title, d.url
"""

_DENSE_SQL = """
WITH hits AS MATERIALIZED (
    SELECT c.chunk_id, c.embedding <=> %(qvec)s::vector AS distance
    FROM chunks c
    WHERE {where}
    ORDER BY c.embedding <=> %(qvec)s::vector
    LIMIT %(k)s
)
SELECT {cols}, 1 - hits.distance AS score
FROM hits
JOIN chunks c ON c.chunk_id = hits.chunk_id
JOIN documents d ON d.doc_id = c.doc_id
ORDER BY hits.distance, c.chunk_id
"""

# OR the query's terms (plainto_tsquery ANDs them): an agent's natural-language query rarely has
# every word in one chunk, and RRF only needs a sensible rank. ts_rank_cd ranks fuller matches up.
_LEXICAL_SQL = """
WITH q AS (
    SELECT replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery AS tsq
)
SELECT {cols}, ts_rank_cd(c.tsv, q.tsq) AS score
FROM q, chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE numnode(q.tsq) > 0 AND c.tsv @@ q.tsq AND {where}
ORDER BY score DESC, c.chunk_id
LIMIT %(k)s
"""

_UPSERT_DOC_SQL = """
INSERT INTO documents (doc_id, source, source_type, url, title, owners, created_at, updated_at,
                       parent_id, acl_principals, content_hash, extra)
VALUES (%(doc_id)s, %(source)s, %(source_type)s, %(url)s, %(title)s, %(owners)s, %(created_at)s,
        %(updated_at)s, %(parent_id)s, %(acl)s, %(content_hash)s, %(extra)s)
ON CONFLICT (doc_id) DO UPDATE SET
    source = EXCLUDED.source, source_type = EXCLUDED.source_type, url = EXCLUDED.url,
    title = EXCLUDED.title, owners = EXCLUDED.owners, created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at, parent_id = EXCLUDED.parent_id,
    acl_principals = EXCLUDED.acl_principals, content_hash = EXCLUDED.content_hash,
    extra = EXCLUDED.extra
"""

_UPSERT_CHUNK_SQL = """
INSERT INTO chunks (chunk_id, doc_id, ordinal, section_path, text, context, embed_text, embedding,
                    source, updated_at, acl_principals)
VALUES (%(chunk_id)s, %(doc_id)s, %(ordinal)s, %(section_path)s, %(text)s, %(context)s,
        %(embed_text)s, %(embedding)s::vector, %(source)s, %(updated_at)s, %(acl)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id, ordinal = EXCLUDED.ordinal, section_path = EXCLUDED.section_path,
    text = EXCLUDED.text, context = EXCLUDED.context, embed_text = EXCLUDED.embed_text,
    embedding = EXCLUDED.embedding, source = EXCLUDED.source, updated_at = EXCLUDED.updated_at,
    acl_principals = EXCLUDED.acl_principals
"""


# --- the store --------------------------------------------------------------------------------


@register(STORE, "postgres")
class PostgresStore:
    """The :class:`~hackstack.store.base.Store` on Postgres 17 + pgvector.

    ``upsert`` needs every chunk's ``embedding`` set (by the embedder). The document's effective
    ACL is authoritative: it is copied onto every chunk it writes. One connection, serialized by a
    lock; the schema is created on first use.
    """

    def __init__(
        self,
        dsn: str,
        *,
        embed_model: str,
        dim: int,
        ingest_user: str | None = None,
        ef_search_min: int = EF_SEARCH_MIN,
    ) -> None:
        render_schema(dim)  # validates dim early
        self.dsn = dsn
        self.embed_model = embed_model
        self.dim = dim
        self.ingest_user = ingest_user
        self.ef_search_min = ef_search_min
        self._conn: psycopg.Connection | None = None
        self._lock = threading.RLock()
        self._ready = False

    @classmethod
    def from_settings(cls, settings: Settings, *, ingest_user: str | None = None) -> PostgresStore:
        return cls(
            settings.store.dsn,
            embed_model=settings.embed.model,
            dim=settings.embed.dim,
            ingest_user=ingest_user,
        )

    # -- lifecycle -----------------------------------------------------------------------------

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
            self._ready = False
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
            self._conn = None
            self._ready = False

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        """Create the schema if needed; raise if the index was built by another embedder."""
        with self._lock:
            conn = self._connection()
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK,))
                self._create_and_check(conn)
            self._ready = True

    def reset(self) -> None:
        """Drop the index (chunks, documents, sync cursors, embedder record) and recreate it."""
        with self._lock:
            conn = self._connection()
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK,))
                conn.execute(f"DROP TABLE IF EXISTS {', '.join(_TABLES)} CASCADE")
                self._create_and_check(conn)
            self._ready = True

    def _create_and_check(self, conn: psycopg.Connection) -> None:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS collection_meta (id boolean PRIMARY KEY DEFAULT true"
            " CHECK (id), embed_model text NOT NULL, dim int NOT NULL,"
            " created_at timestamptz NOT NULL DEFAULT now())"
        )
        meta = conn.execute("SELECT embed_model, dim FROM collection_meta").fetchone()
        if meta is not None:
            if (meta["embed_model"], meta["dim"]) != (self.embed_model, self.dim):
                raise IndexMismatchError(
                    f"the index was built with {meta['embed_model']} ({meta['dim']} dim) but the "
                    f"config says {self.embed_model} ({self.dim} dim). Mixing vectors from "
                    "different models is refused: run `make reset` and re-ingest."
                )
        else:
            col = conn.execute(
                "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute"
                " WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
            ).fetchone()
            if col is not None and col["t"] != f"vector({self.dim})":
                raise IndexMismatchError(
                    f"chunks.embedding is {col['t']} but embed.dim is {self.dim}: run `make reset`"
                )
        conn.execute(render_schema(self.dim))
        if meta is None:
            conn.execute(
                "INSERT INTO collection_meta (embed_model, dim) VALUES (%s, %s)",
                (self.embed_model, self.dim),
            )

    def _ready_conn(self) -> psycopg.Connection:
        conn = self._connection()
        if not self._ready:
            self.ensure_schema()
        return conn

    def _check_dim(self, vec: Sequence[float], what: str) -> None:
        if len(vec) != self.dim:
            raise IndexMismatchError(
                f"{what} has dim {len(vec)} but the index is {self.embed_model} ({self.dim} dim)"
            )

    # -- writes --------------------------------------------------------------------------------

    def upsert(self, doc: RawDoc, chunks: list[Chunk], *, content_hash: str | None = None) -> None:
        """Insert or replace ``doc`` and its chunks; chunks no longer produced are removed.

        ``content_hash`` is recorded as the document's hash (the sync manager passes its doc
        hash); by default it is the SHA-256 of the content.
        """
        for chunk in chunks:
            if chunk.doc_id != doc.doc_id:
                raise ValueError(
                    f"chunk {chunk.chunk_id} belongs to {chunk.doc_id}, not {doc.doc_id}"
                )
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.chunk_id} has no embedding; embed before upsert")
            self._check_dim(chunk.embedding, f"chunk {chunk.chunk_id}")

        acl = effective_principals(doc, ingest_user=self.ingest_user)
        doc_row = {
            "doc_id": doc.doc_id,
            "source": doc.source,
            "source_type": doc.source_type,
            "url": doc.url,
            "title": doc.title,
            "owners": list(doc.owners),
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "parent_id": doc.parent_id,
            "acl": acl,
            "content_hash": content_hash or _content_hash(doc.content),
            "extra": Jsonb(doc.model_dump(mode="json")["extra"]),
        }
        chunk_rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "ordinal": c.ordinal,
                "section_path": list(c.section_path),
                "text": c.text,
                "context": c.context,
                "embed_text": c.embed_text,
                "embedding": vector_literal(c.embedding or []),
                "source": c.source,
                "updated_at": c.updated_at,
                "acl": acl,
            }
            for c in chunks
        ]
        with self._lock:
            conn = self._ready_conn()
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(_UPSERT_DOC_SQL, doc_row)
                cur.execute(
                    "DELETE FROM chunks WHERE doc_id = %s AND NOT (chunk_id = ANY(%s))",
                    (doc.doc_id, [r["chunk_id"] for r in chunk_rows]),
                )
                if chunk_rows:
                    cur.executemany(_UPSERT_CHUNK_SQL, chunk_rows)

    def delete_doc(self, doc_id: str) -> None:
        with self._lock:
            self._ready_conn().execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    # -- incremental sync (hackstack.sync.manager.SyncStore) -----------------------------------

    def doc_hashes(self, source: str) -> dict[str, str | None]:
        with self._lock:
            rows = (
                self._ready_conn()
                .execute("SELECT doc_id, content_hash FROM documents WHERE source = %s", (source,))
                .fetchall()
            )
        return {r["doc_id"]: r["content_hash"] for r in rows}

    def chunk_embeddings(self, doc_id: str) -> list[tuple[str, list[float]]]:
        with self._lock:
            rows = (
                self._ready_conn()
                .execute(
                    "SELECT embed_text, embedding::text AS vec FROM chunks WHERE doc_id = %s",
                    (doc_id,),
                )
                .fetchall()
            )
        return [(r["embed_text"], [float(x) for x in json.loads(r["vec"])]) for r in rows]

    def get_cursor(self, connector: str) -> dict[str, Any] | None:
        with self._lock:
            row = (
                self._ready_conn()
                .execute("SELECT cursor FROM sync_state WHERE connector = %s", (connector,))
                .fetchone()
            )
        return None if row is None or row["cursor"] is None else dict(row["cursor"])

    def set_cursor(self, connector: str, cursor: dict[str, Any]) -> None:
        with self._lock:
            self._ready_conn().execute(
                "INSERT INTO sync_state (connector, cursor, updated_at) VALUES (%s, %s, now())"
                " ON CONFLICT (connector) DO UPDATE SET cursor = EXCLUDED.cursor,"
                " updated_at = EXCLUDED.updated_at",
                (connector, Jsonb(cursor)),
            )

    # -- reads ---------------------------------------------------------------------------------

    def dense(
        self,
        qvec: list[float],
        principals: list[str],
        filters: dict[str, Any] | None,
        k: int,
    ) -> list[SearchHit]:
        """Nearest chunks by cosine similarity, among those ``principals`` may see."""
        self._check_dim(qvec, "query vector")
        where, params = filter_clause(principals, filters)
        if not params["principals"] or k <= 0:
            return []
        params |= {"qvec": vector_literal(qvec), "k": k}
        ef_search = min(max(k, self.ef_search_min), EF_SEARCH_MAX)
        with self._lock:
            conn = self._ready_conn()
            with conn.transaction():
                # Filters apply after the index scan; iterative scans keep it going until k rows
                # pass, and ef_search >= k stops the index from capping the candidate count.
                conn.execute(
                    "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true),"
                    " set_config('hnsw.ef_search', %s, true)",
                    (str(ef_search),),
                )
                rows = conn.execute(
                    _DENSE_SQL.format(where=where, cols=_HIT_COLUMNS), params
                ).fetchall()
        return [_hit(r, "dense") for r in rows]

    def lexical(
        self,
        query: str,
        principals: list[str],
        filters: dict[str, Any] | None,
        k: int,
    ) -> list[SearchHit]:
        """Full-text matches (any query term, english stemming) ranked by ``ts_rank_cd``."""
        where, params = filter_clause(principals, filters)
        if not query.strip() or not params["principals"] or k <= 0:
            return []
        params |= {"query": query, "k": k}
        with self._lock:
            rows = (
                self._ready_conn()
                .execute(_LEXICAL_SQL.format(where=where, cols=_HIT_COLUMNS), params)
                .fetchall()
            )
        return [_hit(r, "lexical") for r in rows]

    # -- maintenance ---------------------------------------------------------------------------

    def count_chunks(self) -> int:
        with self._lock:
            row = self._ready_conn().execute("SELECT count(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def analyze(self) -> None:
        """Refresh planner statistics (after a bulk ingest)."""
        with self._lock:
            self._ready_conn().execute("ANALYZE chunks")


def _hit(row: dict[str, Any], retriever: str) -> SearchHit:
    return SearchHit(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        text=row["text"],
        score=float(row["score"]),
        title=row["title"],
        url=row["url"],
        source=row["source"],
        section_path=list(row["section_path"] or []),
        updated_at=row["updated_at"],
        retriever=retriever,
    )
