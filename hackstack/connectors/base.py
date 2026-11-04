from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from hackstack.models import RawDoc


class ConnectorFailure(BaseModel):
    """One item that could not be ingested. Yielded or recorded instead of crashing the run.

    ``stage`` says where it failed: ``load`` (the connector), ``parse`` or ``chunk``.
    """

    connector: str
    item_id: str
    stage: Literal["load", "parse", "chunk"] = "load"
    error: str
    error_type: str = ""

    @classmethod
    def from_exc(
        cls, connector: str, item_id: str, exc: BaseException, stage: str = "load"
    ) -> ConnectorFailure:
        return cls(
            connector=connector,
            item_id=item_id,
            stage=stage,  # type: ignore[arg-type]
            error=str(exc) or type(exc).__name__,
            error_type=type(exc).__name__,
        )


@runtime_checkable
class Connector(Protocol):
    """Register implementations with ``@register("connector", "<name>")``.

    ``load_all`` yields a :class:`ConnectorFailure` for an item it cannot read, then carries
    on: one bad file or page must not crash a whole run.

    Optional capabilities, detected with ``hasattr`` by the sync manager
    (:mod:`hackstack.sync.manager`):

    * ``poll(since, until) -> Iterator[RawDoc | ConnectorFailure]``: only the items changed in
      ``[since, until)``. Cheap increments; a miss costs freshness, never correctness, because
      the manager hash-checks whatever it gets and a ``--full`` run re-reads everything.
    * ``list_ids() -> Iterator[str]``: every ``doc_id`` the source has right now, without
      loading content. Used for deletion checks, and to find items ``poll`` could not see.
    * ``volatile_extra: frozenset[str]``: ``RawDoc.extra`` keys that change without the item
      changing (a file's mtime); left out of the doc hash.
    """

    name: str

    def configure(self, cfg: dict[str, Any], secrets: dict[str, Any]) -> None: ...

    def validate(self) -> None:
        """Fail fast on a bad token, missing scope or unreachable source."""
        ...

    def load_all(self) -> Iterator[RawDoc | ConnectorFailure]: ...


class PollingConnector(Connector, Protocol):
    def poll(
        self, since: datetime, until: datetime | None = None
    ) -> Iterator[RawDoc | ConnectorFailure]: ...


class ListingConnector(Connector, Protocol):
    def list_ids(self) -> Iterator[str]: ...
