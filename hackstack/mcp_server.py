"""MCP server: the knowledge base's ``search_kb`` and ``get_document`` over stdio.

Claude Desktop and Claude Code start it as a subprocess (``make mcp`` runs it by hand)::

    uv --directory /path/to/hackstack run python -m hackstack.mcp_server

It answers as one user, fixed when it starts: ``--as <id>``, else ``HACKSTACK_AS_USER``, else
the active app's ``default_user``. That user's ACL applies to every call. The identity is never
a tool argument, because then the model could pick whoever can read the most. An unknown user
stops the server at startup instead of quietly searching as nobody.

The tools are the agent's own (``hackstack/agent/tools.py``): the same handlers, ACL and input
checks. Their results are plain text, one numbered block per passage with its title, ``kb://``
source id, date and link, because MCP clients do not all accept ``search_result`` blocks.

stdout carries the protocol, so everything else (logs, warnings) goes to stderr. Claude Desktop
keeps a server's stderr in its ``mcp-server-hackstack.log``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import TextContent, ToolAnnotations
from pydantic import Field

from hackstack.agent.citations import CitationRegistry, display_title
from hackstack.agent.tools import (
    _GET_DOCUMENT_DESCRIPTION,
    _SEARCH_DESCRIPTION,
    GET_DOCUMENT,
    SEARCH_KB,
    TOOL,
    Passages,
    ToolContext,
    ToolInputError,
)
from hackstack.apps import DEFAULT_APPS_DIR, App, load_app
from hackstack.config import Settings
from hackstack.llm.citations import KB_SCHEME
from hackstack.models import SearchHit
from hackstack.registry import get
from hackstack.retrieval.acl import UnknownUserError, principals_for

AS_USER_ENV = "HACKSTACK_AS_USER"
SERVER_NAME = "hackstack"

log = logging.getLogger("hackstack.mcp")


def resolve_user(
    settings: Settings, user: str | None, *, apps_dir: str | Path = DEFAULT_APPS_DIR
) -> tuple[App, str, str, list[str]]:
    """``(app, user, where the user came from, principals)``; raises :class:`UnknownUserError`."""
    if not (Path(apps_dir) / settings.app / "app.yaml").is_file():
        raise UnknownUserError(f"no app folder for app {settings.app!r}, so no users to answer as")
    app = load_app(settings, apps_dir=apps_dir)
    if user:
        source = "argument"
    elif os.environ.get(AS_USER_ENV):
        user, source = os.environ[AS_USER_ENV], AS_USER_ENV
    elif app.default_user:
        user, source = app.default_user, "app default_user"
    else:
        raise UnknownUserError(f"no user: set {AS_USER_ENV} or the app's default_user")
    try:
        principals = principals_for(app.principal(user))
    except KeyError as exc:
        raise UnknownUserError(
            f"unknown user {user!r} (from {source}); known: {', '.join(p.id for p in app.users)}"
        ) from exc
    return app, user, source, principals


class KBServer:
    """The knowledge base as one fixed user sees it. Builds the retriever on first use."""

    def __init__(
        self,
        settings: Settings,
        *,
        user: str | None = None,
        apps_dir: str | Path = DEFAULT_APPS_DIR,
        retriever: Any = None,
        retriever_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.app, self.user, self.user_source, self.principals = resolve_user(
            settings, user, apps_dir=apps_dir
        )
        self._retriever = retriever
        self._factory = retriever_factory or self._default_retriever
        # Every passage shown so far, so get_document can find a document by its title.
        self.seen = CitationRegistry()

    def _default_retriever(self) -> Any:
        from hackstack.retrieval.hybrid import HybridRetriever
        from hackstack.tracing import setup_tracing

        setup_tracing(self.settings)  # no-op unless tracing.enabled; retrieval spans per call
        return HybridRetriever.from_settings(self.settings)

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = self._factory()
        return self._retriever

    def warm_up(self) -> None:
        """Build the retriever and load the reranker model now, on the calling thread.

        ``main`` calls this before serving. On Windows, loading a native library (numpy, torch,
        libpq) on a worker thread hangs while the stdio transport has a read pending on stdin,
        and tool calls run on worker threads. Loading everything up front avoids that, and the
        first search is not slowed by model loading either.
        """
        load = getattr(getattr(self.retriever, "reranker", None), "load", None)
        if callable(load):
            load()

    def _context(self) -> ToolContext:
        return ToolContext(
            principals=self.principals,
            retriever=self.retriever,
            citations=self.seen,
            user=self.user,
            document_chars=4 * self.settings.expand.max_chars,
            settings=self.settings,
            app=self.app,
        )

    def _run(self, name: str, args: dict[str, Any]) -> list[str]:
        try:
            out = get(TOOL, name).handler(args, self._context())
        except ToolInputError as exc:
            raise ToolError(str(exc)) from exc
        return self._render(out)

    def search(self, query: str, source: str | None = None, since: str | None = None) -> list[str]:
        return self._run(SEARCH_KB, {"query": query, "source": source, "since": since})

    def get_document(self, document: str) -> list[str]:
        return self._run(GET_DOCUMENT, {"document": document})

    def _render(self, out: Any) -> list[str]:
        if isinstance(out, str):
            return [out]
        assert isinstance(out, Passages)
        if not out.hits:
            return [out.empty]
        blocks = []
        for number, hit in enumerate(out.hits, start=1):
            self.seen.add(hit)
            blocks.append(_passage(number, hit))
        return blocks


def _passage(number: int, hit: SearchHit) -> str:
    # display_title already carries the section path and "(updated <date>)"
    lines = [f"[{number}] {display_title(hit)}", f"source: {KB_SCHEME}{hit.chunk_id}"]
    if hit.url:
        lines.append(f"url: {hit.url}")
    return "\n".join(lines) + "\n\n" + hit.text


_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


def build_server(kb: KBServer) -> FastMCP:
    """A FastMCP server exposing ``kb``'s two tools."""
    title = kb.app.title if kb.app is not None else "Hackstack"
    mcp = FastMCP(
        SERVER_NAME,
        instructions=(
            f"Search {title}'s knowledge base as {kb.user}. Cite passages by their [n] title "
            "and source id. Results only include documents this user may read."
        ),
        log_level="WARNING",
    )
    logged: list[bool] = []

    def note_client(ctx: Context) -> None:
        if logged:
            return
        logged.append(True)
        params = getattr(ctx.session, "client_params", None)
        if params is not None:
            info = params.clientInfo
            log.warning(
                "client %s %s, protocol %s", info.name, info.version, params.protocolVersion
            )

    async def search_kb(
        query: Annotated[str, Field(description="What to look for, in plain words.")],
        ctx: Context,
        source: Annotated[
            str | None,
            Field(description="Only this source, e.g. local_files, github, slack, notion."),
        ] = None,
        since: Annotated[
            str | None,
            Field(description="Only content updated on or after this date (YYYY-MM-DD)."),
        ] = None,
    ) -> list[TextContent]:
        note_client(ctx)
        blocks = await anyio.to_thread.run_sync(lambda: kb.search(query, source, since))
        return [TextContent(type="text", text=b) for b in blocks]

    async def get_document(
        document: Annotated[
            str,
            Field(description="The document title from a search result, or a kb:// source id."),
        ],
        ctx: Context,
    ) -> list[TextContent]:
        note_client(ctx)
        blocks = await anyio.to_thread.run_sync(lambda: kb.get_document(document))
        return [TextContent(type="text", text=b) for b in blocks]

    mcp.add_tool(
        search_kb,
        name=SEARCH_KB,
        title="Search the knowledge base",
        description=_SEARCH_DESCRIPTION,
        annotations=_READ_ONLY,
        structured_output=False,
    )
    mcp.add_tool(
        get_document,
        name=GET_DOCUMENT,
        title="Read a document",
        description=_GET_DOCUMENT_DESCRIPTION,
        annotations=_READ_ONLY,
        structured_output=False,
    )
    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hackstack.mcp_server", description=__doc__)
    parser.add_argument("--as", dest="as_user", help=f"answer as this user (else ${AS_USER_ENV})")
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(name)s: %(message)s")

    from hackstack.config import load_settings

    try:
        kb = KBServer(load_settings(), user=args.as_user)
    except UnknownUserError as exc:
        print(f"hackstack mcp: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    started = time.monotonic()
    kb.warm_up()
    log.warning(
        "serving %s as %s (from %s), ready in %.1fs",
        kb.settings.app,
        kb.user,
        kb.user_source,
        time.monotonic() - started,
    )
    build_server(kb).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
