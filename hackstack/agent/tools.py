"""Agent tools: the two core tools, and how apps add their own.

A tool is a name, a multi-sentence description, a JSON schema and a handler
``handler(args, ctx) -> Passages | str | list[block]``. Handlers may be sync (run in a worker
thread, so parallel calls really overlap) or async. Return :class:`Passages` for retrieved
text: the loop turns them into numbered, citable ``search_result`` blocks.

Apps add tools in ``apps/<app>/tools.py`` with the :func:`tool` decorator and list them by name
in ``app.yaml``::

    @tool("find_experts", "Find the people who know most about a topic. ...", {...schema...})
    def find_experts(args, ctx): ...

Keep the toolset small and the descriptions long (3-4 sentences): Dust found 16 high-level
tools "produced more errors", and Anthropic's guidance is to describe tools like you would to a
new hire (research doc, *The agent loop*).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hackstack.agent.citations import CitationRegistry
from hackstack.llm.citations import KB_SCHEME
from hackstack.models import SearchHit
from hackstack.registry import available, get, register

if TYPE_CHECKING:
    from hackstack.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)

TOOL = "tool"  # registry kind
DOCUMENT_WINDOW = 10_000  # get_document reads ordinals 0..this


class ToolInputError(ValueError):
    """The model called a tool with arguments it can fix; shown to it as an error result."""


@dataclass
class Passages:
    """Retrieved text, returned to the model as ``search_result`` blocks it can cite.

    ``skip_seen`` drops passages already shown earlier in the conversation (they cost context
    and add nothing); ``empty`` is the plain-text result when nothing is left. ``preamble`` is
    plain text sent before the passages (an app tool's summary, e.g. a ranked list of people);
    it is always sent, even when every passage was shown before.
    """

    hits: list[SearchHit]
    skip_seen: bool = True
    empty: str = "No results found."
    preamble: str = ""


ToolOutput = Passages | str | list[dict[str, Any]]


@dataclass
class ToolContext:
    """What a tool may use for one request: who is asking, and the kit's components."""

    principals: list[str]
    retriever: HybridRetriever | Any = None
    citations: CitationRegistry = field(default_factory=CitationRegistry)
    user: str = ""
    top_n: int | None = None  # passages per search (default: rerank.top_n)
    document_chars: int = 8_000  # get_document's budget
    settings: Any = None
    app: Any = None
    extra: dict[str, Any] = field(default_factory=dict)  # app-specific state


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]

    def definition(self) -> dict[str, Any]:
        """The Messages-API tool definition (adapters translate it)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def tool(name: str, description: str, input_schema: dict[str, Any]) -> Callable:
    """Decorator: register ``handler(args, ctx)`` as an agent tool named ``name``."""

    def decorator(handler: Callable) -> Callable:
        register(TOOL, name)(Tool(name, description, input_schema, handler))
        return handler

    return decorator


def load_tools(names: list[str], *, app_root: Path | None = None) -> list[Tool]:
    """The tools ``names`` lists, importing ``<app_root>/tools.py`` first if it exists.

    A listed tool nobody registered is skipped with a warning, so an app can list tools that a
    later change will add without breaking ``ask``.
    """
    if app_root is not None and (app_root / "tools.py").is_file():
        _import_app_tools(app_root)
    out = []
    for name in names:
        if name not in available(TOOL):
            log.warning("tool %r is listed but not registered; skipping it", name)
            continue
        out.append(get(TOOL, name))
    return out


def _import_app_tools(app_root: Path) -> None:
    module_name = f"hackstack_app_tools_{app_root.name}"
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, app_root / "tools.py")
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


# --- core tools --------------------------------------------------------------------------------

SEARCH_KB = "search_kb"
GET_DOCUMENT = "get_document"

_SEARCH_DESCRIPTION = """\
Search the company knowledge base and return the most relevant passages, each with its \
document title, section and last-updated date. Only documents the person asking may read are \
searched. Use it before answering any question about the company, and when a question has \
several parts, run one search per part in the same turn. Write the query as a short \
natural-language question or phrase, not a single keyword. Optional filters: `source` limits \
results to one source (for example local_files, github, slack, notion), and `since` \
(YYYY-MM-DD) to content updated on or after that date, which suits "what changed recently" \
questions. "No results found." means nothing matched: the information may not exist, or this \
person may not be allowed to see it."""

_GET_DOCUMENT_DESCRIPTION = """\
Read a whole document when a search passage is not enough, for example to see a complete \
policy, a full list, or the context around a passage. Pass the document title exactly as it \
appears at the start of a search result title (before any " > "), or a source id such as \
kb://local_files:handbook/pto.md#2. Returns the document's passages in order, up to a few \
thousand characters. Only documents the person asking may read are returned, so a document \
you cannot get may exist but be restricted."""


def _search_kb(args: dict[str, Any], ctx: ToolContext) -> Passages:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolInputError("search_kb needs a non-empty `query`.")
    source = args.get("source") or None
    since = args.get("since") or None
    if since is not None:
        try:
            date.fromisoformat(str(since)[:10])
        except ValueError as exc:
            raise ToolInputError(f"`since` must be a date like 2026-11-01, got {since!r}.") from exc
        since = str(since)[:10]
    filters = {"source": source, "since": since}
    hits = ctx.retriever.search(query, ctx.principals, filters, top_n=ctx.top_n)
    return Passages(hits=list(hits))


def _get_document(args: dict[str, Any], ctx: ToolContext) -> Passages:
    name = str(args.get("document") or args.get("doc_id") or "").strip()
    if not name:
        raise ToolInputError("get_document needs `document`: a title or a source id.")
    doc_id = ctx.citations.find_doc(name)
    if doc_id is None:
        key = name.removeprefix(KB_SCHEME)
        doc_id = key.rsplit("#", 1)[0] if "#" in key else key
    hits = ctx.retriever.store.neighbors([(doc_id, 0, DOCUMENT_WINDOW)], ctx.principals)
    budget = ctx.document_chars
    kept: list[SearchHit] = []
    for h in hits:  # ordered by position; the store applied the ACL
        if kept and budget - len(h.text) < 0:
            last = kept[-1]
            kept[-1] = last.model_copy(update={"text": last.text + "\n\n[document truncated]"})
            break
        kept.append(h)
        budget -= len(h.text)
    return Passages(
        hits=kept,
        skip_seen=False,
        empty=(
            f"No document matching {name!r} that this person can read. Use a title from a "
            "search result, or search first."
        ),
    )


tool(
    SEARCH_KB,
    _SEARCH_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "source": {
                "type": "string",
                "description": "Only this source, e.g. local_files, github, slack, notion.",
            },
            "since": {
                "type": "string",
                "description": "Only content updated on or after this date (YYYY-MM-DD).",
            },
        },
        "required": ["query"],
    },
)(_search_kb)

tool(
    GET_DOCUMENT,
    _GET_DOCUMENT_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "document": {
                "type": "string",
                "description": "The document title from a search result, or a kb:// source id.",
            }
        },
        "required": ["document"],
    },
)(_get_document)

CORE_TOOLS = [SEARCH_KB, GET_DOCUMENT]
