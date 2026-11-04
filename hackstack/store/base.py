from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hackstack.models import Chunk, RawDoc, SearchHit


@runtime_checkable
class Store(Protocol):
    """Register implementations with ``@register("store", "<name>")``.

    Both retrievers must apply the permission filter themselves (inside the query), and must
    return nothing for principals that match no chunk ACL: fail closed.
    """

    def upsert(self, doc: RawDoc, chunks: list[Chunk]) -> None: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def dense(
        self,
        qvec: list[float],
        principals: list[str],
        filters: dict[str, Any] | None,
        k: int,
    ) -> list[SearchHit]: ...

    def lexical(
        self,
        query: str,
        principals: list[str],
        filters: dict[str, Any] | None,
        k: int,
    ) -> list[SearchHit]: ...
