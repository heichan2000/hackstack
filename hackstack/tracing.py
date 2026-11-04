"""OpenTelemetry tracing: one trace per request, a span per stage, exported to Phoenix (optional).

One agent turn (``hackstack ask``, ``POST /chat``, the MCP server) is one trace::

    invoke_agent {app}                         AGENT     gen_ai.conversation.id, mode, usage
    ├── chat {model}                           LLM       tokens in/out, finish reason, iteration
    ├── execute_tool search_kb                 TOOL      call id, passages returned
    │   └── retrieval                          RETRIEVER document ids + scores, principals count
    │       ├── embeddings {model}             EMBEDDING dimensions
    │       ├── dense / lexical                          window, hits
    │       ├── rrf                                      k, fused count
    │       ├── rerank                         RERANKER  model, in -> out
    │       └── expand                                   radius, out
    └── chat {model}                                     (the answer)

A real ``/chat`` trace in Phoenix: ``docs/images/phoenix-trace.png`` (FastAPI adds its own
``POST /chat`` span above ``invoke_agent`` once ``opentelemetry-api`` is installed).

``single_shot`` is ``invoke_agent -> retrieval -> chat``. There is no query-rewrite span: the
agent writes its own search queries (they are the ``execute_tool`` spans), and single-shot
searches with the question as asked. ``hackstack search`` traces just the ``retrieval`` subtree.

Names and attributes live in :mod:`hackstack.semconv` (GenAI semconv pinned to v1.41.0, plus
``openinference.span.kind`` so Phoenix shows each kind natively).

**Off by default, and free when off.** Instrumented code only uses ``opentelemetry-api``, whose
default provider hands out non-recording spans. :func:`setup_tracing` installs the SDK and an
OTLP/HTTP exporter only when ``tracing.enabled`` is true; both live in the ``tracing`` extra
(``uv sync --extra tracing``, included in ``make setup``). Enabled without the extra, it warns
and the kit runs untraced; an unreachable collector only costs background export errors.

**Content is opt-in.** Prompts, questions, search queries, tool arguments and answers go on spans
only with ``tracing.capture_content: true`` (``HACKSTACK_TRACING__CAPTURE_CONTENT``): spans would
otherwise carry whatever PII the documents and questions hold. Ids, counts, scores, models and
timings are always recorded.

Async generators: the agent's spans cross ``yield`` points (an SSE consumer may stop at any of
them), so they are started detached and parented explicitly; only code without a ``yield``
inside it (tool calls, retrieval in its worker thread) makes a span *current*.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from hackstack import semconv as sc

if TYPE_CHECKING:
    from hackstack.config import Settings

log = logging.getLogger(__name__)

TRACER_NAME = "hackstack"
CONTENT_LIMIT = 16_000  # characters of captured content per attribute
EXPORT_TIMEOUT_S = 5.0  # a down collector must not hold up `hackstack ask` exiting for long

_provider: Any = None  # the provider spans come from; None -> the global (no-op unless set)
_installed: Any = None  # what setup_tracing installed, so a second call is a no-op
_capture = False


# --- setup ---------------------------------------------------------------------------------------


def build_provider(settings: Settings, *, exporter: Any = None) -> Any:
    """An SDK ``TracerProvider`` for ``settings``: resource tags + OTLP/HTTP batch export.

    Raises ``ImportError`` without the ``tracing`` extra.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from hackstack.config import config_hash

    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        from hackstack.config import load_secrets

        exporter = OTLPSpanExporter(
            endpoint=settings.tracing.endpoint,
            headers=exporter_headers(load_secrets()),
            timeout=EXPORT_TIMEOUT_S,
        )
    resource = Resource.create(
        {
            sc.SERVICE_NAME: TRACER_NAME,
            sc.SERVICE_VERSION: _version(),
            sc.OI_PROJECT_NAME: settings.tracing.project,
            sc.HS_CONFIG_HASH: config_hash(settings),
            sc.HS_PROFILE: settings.profile,
            sc.HS_APP: settings.app,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def exporter_headers(secrets: Any) -> dict[str, str]:
    """``Authorization: Bearer`` for a Phoenix with auth on (``PHOENIX_API_KEY`` in ``.env``)."""
    key = getattr(secrets, "phoenix_api_key", None)
    return {"authorization": f"Bearer {key.get_secret_value()}"} if key else {}


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("hackstack")
    except PackageNotFoundError:  # pragma: no cover - running from a bare checkout
        return "0+unknown"


def setup_tracing(settings: Settings) -> Any:
    """Install tracing if ``tracing.enabled``; returns the provider, or None when off.

    Idempotent: the CLI, the API and :class:`~hackstack.agent.runtime.Agent` all call it.
    """
    global _installed
    if not settings.tracing.enabled:
        return None
    if _installed is not None:
        return _installed
    try:
        provider = build_provider(settings)
    except ImportError as exc:
        log.warning(
            "tracing.enabled is true but the tracing extra is not installed (%s); running "
            "untraced. Install it with `uv sync --extra tracing` (or `make setup`).",
            exc,
        )
        return None
    trace.set_tracer_provider(provider)
    use_provider(provider, capture_content=settings.tracing.capture_content)
    _installed = provider
    log.info("tracing to %s (project %s)", settings.tracing.endpoint, settings.tracing.project)
    return provider


def use_provider(provider: Any, *, capture_content: bool = False) -> None:
    """Send hackstack's spans to ``provider`` (tests pass one with an in-memory exporter)."""
    global _provider, _capture
    _provider = provider
    _capture = capture_content


def reset() -> None:
    """Forget the provider (back to the no-op global). For tests."""
    global _provider, _installed, _capture
    _provider = None
    _installed = None
    _capture = False


def is_enabled() -> bool:
    return _provider is not None


def capturing_content() -> bool:
    return _capture


def force_flush() -> None:
    """Export buffered spans now (a short-lived CLI run would otherwise rely on atexit)."""
    if _provider is not None and hasattr(_provider, "force_flush"):
        _provider.force_flush(timeout_millis=int(EXPORT_TIMEOUT_S * 1000))


def tracer() -> trace.Tracer:
    source = _provider if _provider is not None else trace.get_tracer_provider()
    return source.get_tracer(TRACER_NAME, schema_url=None)


# --- spans ---------------------------------------------------------------------------------------


def _clean(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (attributes or {}).items() if v is not None}


def start_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    parent: Span | None = None,
) -> Span:
    """A detached span (not made current): for async generators. The caller must ``end()`` it."""
    context = trace.set_span_in_context(parent) if parent is not None else None
    return tracer().start_span(name, context=context, kind=kind, attributes=_clean(attributes))


@contextmanager
def span(
    name: str, attributes: Mapping[str, Any] | None = None, *, kind: SpanKind = SpanKind.INTERNAL
) -> Iterator[Span]:
    """A current span for code without a ``yield`` inside; an exception marks it as an error."""
    with tracer().start_as_current_span(
        name,
        kind=kind,
        attributes=_clean(attributes),
        record_exception=False,
        set_status_on_exception=False,
    ) as s:
        try:
            yield s
        except BaseException as exc:
            fail(s, exc)
            raise


@contextmanager
def use(parent: Span) -> Iterator[None]:
    """Make ``parent`` current for a ``yield``-free block, so spans started inside nest under it."""
    with trace.use_span(parent, end_on_exit=False, record_exception=False,
                        set_status_on_exception=False):  # fmt: skip
        yield


def fail(s: Span, exc: BaseException | None = None, *, error_type: str | None = None) -> None:
    """Mark ``s`` as failed: ``error.type`` (the exception class by default) and ERROR status."""
    kind = error_type or (type(exc).__name__ if exc is not None else "error")
    s.set_attribute(sc.ERROR_TYPE, kind)
    if exc is not None:
        s.record_exception(exc)
    s.set_status(Status(StatusCode.ERROR, str(exc) if exc is not None else kind))


def set_attributes(s: Span, attributes: Mapping[str, Any]) -> None:
    for key, value in _clean(attributes).items():
        s.set_attribute(key, value)


def content(s: Span, key: str, value: Any) -> None:
    """Put prompt/query/answer text on ``s``, only when ``tracing.capture_content`` is on."""
    if not _capture or value is None:
        return
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    s.set_attribute(key, text[:CONTENT_LIMIT])


def documents(hits: Any) -> str:
    """``gen_ai.retrieval.documents``: ids and scores only, as the JSON the semconv allows."""
    return json.dumps([{"id": h.chunk_id, "score": round(float(h.score), 4)} for h in hits])


def server_attributes(base_url: str | None) -> dict[str, Any]:
    if not base_url:
        return {}
    parsed = urlparse(base_url)
    return {sc.SERVER_ADDRESS: parsed.hostname, sc.SERVER_PORT: parsed.port}
