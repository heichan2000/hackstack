"""The ingest pipeline: connector -> sanitize -> parse -> chunk -> embed -> store.

Every stage is an interface, so any piece can be swapped from ``hackstack.yaml``. Per-item
problems (a file that will not load, parse or chunk) become :class:`ConnectorFailure` records
in the :class:`IngestReport`; the run carries on. Embedder and store errors are infrastructure
problems and do raise: retrying every remaining document against a dead store helps nobody.

Without an embedder and a store the run is a dry run: it parses and chunks, and counts.
"""

from __future__ import annotations

import contextlib
import importlib
import time
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from hackstack.chunking.base import Chunker
from hackstack.connectors.base import Connector, ConnectorFailure
from hackstack.embed.base import Embedder
from hackstack.enrich.contextual import ContextualEnricher, EnrichStats
from hackstack.models import Chunk, RawDoc
from hackstack.parsing.base import Parser
from hackstack.parsing.sanitize import sanitize_content, sanitize_text
from hackstack.registry import CHUNKER, CONNECTOR, EMBEDDER, PARSER, STORE, get
from hackstack.store.base import Store

# Where a component named in config lives, if it is not registered yet: hackstack.<pkg>.<name>
_PACKAGES = {
    CONNECTOR: "connectors",
    PARSER: "parsing",
    CHUNKER: "chunking",
    EMBEDDER: "embed",
    STORE: "store",
}


class IngestReport(BaseModel):
    connector: str
    docs: int = 0
    chunks: int = 0
    failures: list[ConnectorFailure] = Field(default_factory=list)
    dry_run: bool = False
    seconds: float = 0.0
    chunks_enriched: int = 0  # chunks given a contextual-retrieval blurb (enrich.contextual)
    enrich: EnrichStats | None = None  # calls, failures, token usage; None when it is off


class Ingestor:
    def __init__(
        self,
        connector: Connector,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder | None = None,
        store: Store | None = None,
        *,
        enricher: ContextualEnricher | None = None,
        batch_size: int = 64,
    ) -> None:
        if (embedder is None) != (store is None):
            raise ValueError("pass both an embedder and a store, or neither (dry run)")
        if enricher is not None and store is None:
            raise ValueError("a dry run does not enrich: it would spend LLM calls for nothing")
        self.connector = connector
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.enricher = enricher
        self.enrich_stats = EnrichStats()
        self.batch_size = batch_size

    @property
    def dry_run(self) -> bool:
        return self.store is None

    def run(self) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport(connector=self.connector.name, dry_run=self.dry_run)
        self.enrich_stats = EnrichStats()
        for item in self.connector.load_all():
            if isinstance(item, ConnectorFailure):
                report.failures.append(item)
                continue
            chunks = self.prepare(item)
            if isinstance(chunks, ConnectorFailure):
                report.failures.append(chunks)
                continue
            if self.store is not None:
                self.embed(chunks)
                self.store.upsert(item, chunks)
            report.docs += 1
            report.chunks += len(chunks)
        analyze = getattr(self.store, "analyze", None)
        if report.docs and callable(analyze):
            analyze()  # a bulk load leaves planner statistics stale
        self.fill_enrich(report)
        report.seconds = round(time.perf_counter() - started, 3)
        return report

    def fill_enrich(self, report: IngestReport) -> None:
        """Copy this run's enrichment totals onto ``report``."""
        if self.enricher is not None:
            report.enrich = self.enrich_stats.model_copy(deep=True)
            report.chunks_enriched = self.enrich_stats.enriched

    def close(self) -> None:
        """Release the store's connection, if it holds one."""
        close = getattr(self.store, "close", None)
        if callable(close):
            close()

    def prepare(self, raw: RawDoc) -> list[Chunk] | ConnectorFailure:
        """Sanitize, parse, chunk and (if on) enrich one document.

        A parse or chunk failure is returned, not raised. A failure of the enrichment LLM
        itself raises, like the embedder's: it is infrastructure, not this document.
        """
        stage = "parse"
        try:
            content = sanitize_content(raw.content, raw.mime)
            parsed = self.parser.parse(content, raw.mime)
            parsed = parsed.model_copy(update={"markdown": sanitize_text(parsed.markdown)})
            stage = "chunk"
            chunks = self.chunker.chunk(parsed, raw)
        except Exception as exc:
            return ConnectorFailure.from_exc(self.connector.name, raw.doc_id, exc, stage=stage)
        if self.enricher is not None and chunks:
            self.enrich_stats.add(self.enricher.enrich(raw, parsed, chunks))
        return chunks

    def embed(self, chunks: list[Chunk]) -> None:
        """Set ``chunk.embedding`` on every chunk, in batches."""
        assert self.embedder is not None
        for batch in _batches(chunks, self.batch_size):
            vectors = self.embedder.embed_documents([c.embed_text for c in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(batch)}")
            for chunk, vec in zip(batch, vectors, strict=True):
                chunk.embedding = vec


def _batches(items: list[Chunk], n: int) -> Iterator[list[Chunk]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# --- building components from settings --------------------------------------------------------


def resolve(kind: str, name: str) -> Any:
    """The component registered as ``kind``/``name``, importing ``hackstack.<pkg>.<name>`` first
    if nothing has registered it yet."""
    try:
        return get(kind, name)
    except KeyError:
        package = _PACKAGES.get(kind)
        if package is None:
            raise
        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(f"hackstack.{package}.{name}")
        return get(kind, name)


def build(kind: str, name: str, settings: Any) -> Any:
    """Instantiate a component: ``cls.from_settings(settings)`` if it has one, else ``cls()``."""
    cls = resolve(kind, name)
    factory = getattr(cls, "from_settings", None)
    return factory(settings) if callable(factory) else cls()


def ingestor_from_settings(
    settings: Any,
    connector_name: str,
    secrets: dict[str, Any] | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    path: str | None = None,
    dry_run: bool = False,
) -> Ingestor:
    """Wire an :class:`Ingestor` for one connector.

    ``cfg`` is the connector's config (e.g. from ``load_app(settings).connectors``); by default
    ``settings.connectors[connector_name]``. ``path`` overrides ``cfg["path"]``.
    """
    import hackstack.chunking.markdown  # noqa: F401 - register built-ins
    import hackstack.connectors.local_files  # noqa: F401
    import hackstack.parsing.router  # noqa: F401

    cfg = dict(cfg if cfg is not None else settings.connectors.get(connector_name) or {})
    if path is not None:
        cfg["path"] = path
    connector = resolve(CONNECTOR, connector_name)()
    connector.configure(cfg, secrets or {})
    connector.validate()

    embedder = store = enricher = None
    if not dry_run:
        embedder = build(EMBEDDER, settings.embed.provider, settings)
        store = build(STORE, settings.store.provider, settings)
        if settings.enrich.contextual:
            enricher = ContextualEnricher.from_settings(settings)
    return Ingestor(
        connector=connector,
        parser=build(PARSER, "router", settings),
        chunker=build(CHUNKER, "markdown", settings),
        embedder=embedder,
        store=store,
        enricher=enricher,
    )


def ingest_app(
    settings: Any,
    *,
    connectors: list[str] | None = None,
    path: str | None = None,
    dry_run: bool = False,
    full: bool = False,
    apps_dir: Any = None,
) -> list[Any]:
    """Ingest the active app's sources (``hackstack ingest`` and ``POST /ingest``).

    Connector configs come from the app's ``app.yaml``, overridden by ``hackstack.yaml``; with no
    app folder, ``hackstack.yaml``'s ``connectors:`` alone. Stores that keep sync state run
    incrementally (:class:`~hackstack.sync.manager.SyncManager`). Returns one report per
    connector; raises ``KeyError`` / ``ValueError`` for bad config.
    """
    from pathlib import Path

    from hackstack.apps import DEFAULT_APPS_DIR, load_app
    from hackstack.config import load_secrets
    from hackstack.sync.manager import SyncManager, SyncStore

    apps_dir = Path(apps_dir if apps_dir is not None else DEFAULT_APPS_DIR)
    if (apps_dir / settings.app / "app.yaml").is_file():
        configs = load_app(settings, apps_dir=apps_dir).connectors
    else:
        configs = dict(settings.connectors)
    names = connectors or list(configs) or ["local_files"]
    secrets = {
        k: (v.get_secret_value() if hasattr(v, "get_secret_value") else v)
        for k, v in load_secrets().model_dump().items()
        if v is not None
    }
    reports = []
    for name in names:
        ingestor = ingestor_from_settings(
            settings, name, secrets, cfg=configs.get(name), path=path, dry_run=dry_run
        )
        try:
            if isinstance(ingestor.store, SyncStore):
                reports.append(SyncManager(ingestor).run(full=full))
            else:
                reports.append(ingestor.run())
        finally:
            ingestor.close()
    return reports
