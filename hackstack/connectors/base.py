from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from hackstack.models import RawDoc


@runtime_checkable
class Connector(Protocol):
    """Register implementations with ``@register("connector", "<name>")``.

    Optional capabilities, detected with ``hasattr``:
    ``poll(since: datetime) -> Iterator[RawDoc]`` for cheap increments and
    ``list_ids() -> Iterator[str]`` for deletion checks.
    """

    name: str

    def configure(self, cfg: dict[str, Any], secrets: dict[str, Any]) -> None: ...

    def validate(self) -> None:
        """Fail fast on a bad token, missing scope or unreachable source."""
        ...

    def load_all(self) -> Iterator[RawDoc]: ...


class PollingConnector(Connector, Protocol):
    def poll(self, since: datetime) -> Iterator[RawDoc]: ...


class ListingConnector(Connector, Protocol):
    def list_ids(self) -> Iterator[str]: ...
