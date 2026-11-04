"""MCP client: tools from external MCP servers, as agent tools.

List servers in ``hackstack.yaml`` (the same shape as Claude Desktop's ``mcpServers``)::

    mcp_servers:
      github:
        command: npx
        args: ["-y", "@modelcontextprotocol/server-github"]
        env: {GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"}   # filled from env / .env
        tools: [search_issues]                                   # optional allowlist

At startup :class:`MCPClient` starts each server over stdio, lists its tools and turns each one
into a :class:`~hackstack.agent.tools.Tool` named ``<server>__<tool>`` (``github__search_issues``),
so the model can tell the kit's tools from a server's and two servers never collide.

Results: text output becomes :class:`~hackstack.agent.tools.Passages`, so the model can cite it
like any search result. That happens **only when every block can be wrapped** (text, or an
embedded text resource): a ``tool_result`` holding a ``search_result`` may hold nothing else, so a
result with an image is passed through as plain text and image blocks instead. A server's
``isError`` result becomes an error ``tool_result`` the model can react to.

The sessions live on one background thread with its own event loop, because an MCP session is
tied to the task that opened it while the agent may run in any loop (``asyncio.run`` in the CLI,
uvicorn's loop in the API). Tool handlers hop onto that loop and back. A server that fails to
start is skipped with a warning; the agent still runs without its tools.

Spec note: pinned to ``mcp>=1.27,<2`` over stdio, the path the research doc picked while the v2
SDK and the 2026-07-28 stateless protocol settle (*MCP*, research doc).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from hackstack.agent.tools import Passages, Tool, ToolContext, ToolInputError, ToolOutput
from hackstack.config import MCPServerSettings
from hackstack.models import SearchHit

log = logging.getLogger(__name__)

SEPARATOR = "__"
MAX_NAME = 64  # Anthropic and OpenAI tool names: ^[a-zA-Z0-9_-]{1,64}$
_INVALID = re.compile(r"[^A-Za-z0-9_-]")
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MCPToolError(ToolInputError):
    """The server reported the call failed (``isError``); shown to the model as an error."""


def prefixed_name(server: str, tool: str) -> str:
    """``<server>__<tool>``, made a valid LLM tool name (at most 64 chars, stable, unique)."""
    name = _INVALID.sub("_", f"{server}{SEPARATOR}{tool}")
    if len(name) <= MAX_NAME:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{name[: MAX_NAME - 9]}_{digest}"


def expand_env(env: dict[str, str], *, env_file: str | Path | None = ".env") -> dict[str, str]:
    """``env`` with ``${VAR}`` filled from the environment, then ``.env``.

    A variable whose placeholder cannot be filled is left out (with a warning) rather than
    passed as an empty string, so a server sees "no token" instead of a bogus one.
    """
    file_values: dict[str, str | None] = {}
    if env_file is not None and Path(env_file).is_file():
        file_values = dotenv_values(env_file)
    out: dict[str, str] = {}
    for key, value in env.items():
        found = {v: os.environ.get(v) or file_values.get(v) for v in _PLACEHOLDER.findall(value)}
        missing = [v for v, val in found.items() if not val]
        if missing:
            log.warning("MCP env %s: %s not set; leaving it out", key, ", ".join(missing))
            continue
        filled = value
        for var, val in found.items():
            filled = filled.replace("${" + var + "}", val or "")
        out[key] = filled
    return out


class MCPClient:
    """Connections to the configured MCP servers and the agent tools they provide.

    ``start()`` returns the tools (also kept in ``tools``); ``close()`` stops the servers.
    """

    def __init__(
        self,
        servers: dict[str, MCPServerSettings],
        *,
        startup_timeout: float = 30.0,
        env_file: str | Path | None = ".env",
    ) -> None:
        self.servers = servers
        self.startup_timeout = startup_timeout
        self.env_file = env_file
        self.tools: list[Tool] = []
        self._sessions: dict[str, Any] = {}
        self._serving: dict[str, concurrent.futures.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None

    # --- lifecycle -----------------------------------------------------------------------------

    def start(self) -> list[Tool]:
        enabled = {name: cfg for name, cfg in self.servers.items() if cfg.enabled}
        if not enabled:
            return []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="hackstack-mcp", daemon=True
        )
        self._thread.start()
        self._stop = self._submit(_make_event()).result()

        ready: dict[str, concurrent.futures.Future] = {}
        for name, cfg in enabled.items():
            ready[name] = concurrent.futures.Future()
            self._serving[name] = self._submit(self._serve(name, cfg, ready[name]))

        seen: set[str] = set()
        for name, cfg in enabled.items():
            try:
                listed = ready[name].result(timeout=self.startup_timeout)
            except concurrent.futures.TimeoutError:
                log.warning(
                    "MCP server %r did not start within %.0fs; skipping its tools",
                    name,
                    self.startup_timeout,
                )
                self._serving[name].cancel()
                continue
            except Exception as exc:
                log.warning("MCP server %r failed to start (%s); skipping its tools", name, exc)
                continue
            for mcp_tool in listed:
                if cfg.tools is not None and mcp_tool.name not in cfg.tools:
                    continue
                t = self._tool(name, cfg, mcp_tool)
                if t.name in seen:
                    log.warning("MCP tool name %r is taken; skipping %s", t.name, mcp_tool.name)
                    continue
                seen.add(t.name)
                self.tools.append(t)
            if cfg.tools is not None:
                missing = set(cfg.tools) - {m.name for m in listed}
                if missing:
                    log.warning("MCP server %r has no tool(s) %s", name, ", ".join(sorted(missing)))
        return self.tools

    def close(self) -> None:
        """Stop every server and the background loop. Safe to call twice."""
        loop, self._loop = self._loop, None
        if loop is None:
            return
        if self._stop is not None:
            loop.call_soon_threadsafe(self._stop.set)
        for fut in self._serving.values():
            with contextlib.suppress(Exception):  # already logged; shutting down regardless
                fut.result(timeout=5)
        self._serving.clear()
        self._sessions.clear()
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not loop.is_running():
            loop.close()

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- internals -----------------------------------------------------------------------------

    def _submit(self, coro: Any) -> concurrent.futures.Future:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _serve(
        self, name: str, cfg: MCPServerSettings, ready: concurrent.futures.Future
    ) -> None:
        """Open the session, report its tools, then hold it open until ``close()``."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=cfg.command,
            args=list(cfg.args),
            env=expand_env(cfg.env, env_file=self.env_file) or None,
            cwd=cfg.cwd,
        )
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                listed = await _list_tools(session)
                self._sessions[name] = session
                ready.set_result(listed)
                assert self._stop is not None
                await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            elif not isinstance(exc, asyncio.CancelledError):
                log.warning("MCP server %r stopped: %s", name, exc)
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._sessions.pop(name, None)

    async def _call_on_loop(self, server: str, tool: str, args: dict[str, Any], timeout: float):
        session = self._sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server {server!r} is not connected")
        return await asyncio.wait_for(session.call_tool(tool, args), timeout)

    def _tool(self, server: str, cfg: MCPServerSettings, mcp_tool: Any) -> Tool:
        name = prefixed_name(server, mcp_tool.name)
        original = mcp_tool.name
        summary = (mcp_tool.description or f"The {original} tool.").strip()
        description = f"{summary}\n\n(From MCP server {server!r}; live, not the knowledge base.)"
        schema = dict(mcp_tool.inputSchema or {})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})

        async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
            if self._loop is None:
                raise RuntimeError(f"MCP server {server!r} is not connected")
            fut = self._submit(self._call_on_loop(server, original, args, cfg.timeout))
            result = await asyncio.wrap_future(fut)
            return to_output(server, original, result)

        return Tool(name=name, description=description, input_schema=schema, handler=handler)


async def _make_event() -> asyncio.Event:
    return asyncio.Event()


async def _list_tools(session: Any) -> list[Any]:
    tools: list[Any] = []
    cursor = None
    while True:
        page = await session.list_tools(cursor) if cursor else await session.list_tools()
        tools.extend(page.tools)
        cursor = getattr(page, "nextCursor", None)
        if not cursor:
            return tools


# --- results -----------------------------------------------------------------------------------


def to_output(server: str, tool: str, result: Any) -> ToolOutput:
    """An MCP ``CallToolResult`` as agent tool output (see the module docstring)."""
    from mcp import types

    blocks = list(result.content or [])
    if result.isError:
        text = " ".join(b.text for b in blocks if isinstance(b, types.TextContent)).strip()
        raise MCPToolError(text or f"{tool} on MCP server {server!r} reported an error.")
    if not blocks and result.structuredContent is not None:
        blocks = [types.TextContent(type="text", text=json.dumps(result.structuredContent))]
    if not blocks:
        return "The tool returned no content."

    passages: list[tuple[str, str | None]] = []
    for b in blocks:
        if isinstance(b, types.TextContent) and b.text.strip():
            passages.append((b.text, None))
        elif isinstance(b, types.EmbeddedResource) and isinstance(
            b.resource, types.TextResourceContents
        ):
            passages.append((b.resource.text, str(b.resource.uri)))
        else:
            break
    else:
        call_id = uuid.uuid4().hex[:8]
        doc_id = f"mcp:{server}/{tool}@{call_id}"
        hits = [
            SearchHit(
                chunk_id=f"{doc_id}#{i}",
                doc_id=doc_id,
                text=text,
                score=1.0,
                title=f"{server} {tool}" + (f" ({uri})" if uri else ""),
                url=uri,
                source=f"mcp:{server}",
                retriever="mcp",
            )
            for i, (text, uri) in enumerate(passages)
        ]
        return Passages(hits=hits, skip_seen=False, empty="The tool returned no content.")

    return [_plain_block(b) for b in blocks]


def _plain_block(b: Any) -> dict[str, Any]:
    from mcp import types

    if isinstance(b, types.TextContent):
        return {"type": "text", "text": b.text}
    if isinstance(b, types.ImageContent):
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": b.mimeType, "data": b.data},
        }
    if isinstance(b, types.EmbeddedResource):
        r = b.resource
        if isinstance(r, types.TextResourceContents):
            return {"type": "text", "text": f"{r.uri}:\n{r.text}"}
        return {"type": "text", "text": f"[binary resource {r.uri} ({r.mimeType}) omitted]"}
    if isinstance(b, types.ResourceLink):
        return {"type": "text", "text": f"[resource link: {b.name} {b.uri}]"}
    kind = getattr(b, "type", type(b).__name__)
    mime = getattr(b, "mimeType", "")
    return {"type": "text", "text": f"[{kind} content{' ' + mime if mime else ''} omitted]"}
