"""A tiny stdio MCP server for the MCP client tests.

Run as ``python tests/fixtures/mcp_fixture_server.py``.

Offline and dependency-free beyond ``mcp``. Each tool exercises one way a result can come back.
"""

from __future__ import annotations

import base64

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import EmbeddedResource, ImageContent, TextContent, TextResourceContents

server = FastMCP("fixture")

# A 1x1 transparent PNG.
PIXEL = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@server.tool()
def two_notes() -> list[TextContent]:
    """Return two text blocks."""
    return [
        TextContent(type="text", text="First note: the deploy freeze starts Friday."),
        TextContent(type="text", text="Second note: hotfixes need two approvals."),
    ]


@server.tool()
def readme() -> list[EmbeddedResource]:
    """Return an embedded text resource."""
    return [
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="file:///repo/README.md", mimeType="text/markdown", text="# Repo\nHello."
            ),
        )
    ]


@server.tool()
def chart() -> list[TextContent | ImageContent]:
    """Return text plus an image: cannot be wrapped as search results."""
    return [
        TextContent(type="text", text="Chart attached."),
        ImageContent(type="image", data=PIXEL, mimeType="image/png"),
    ]


@server.tool()
def broken() -> str:
    """Always fails."""
    raise ToolError("the backend is down")


@server.tool(name="weird.name/with spaces")
def weird() -> str:
    """A tool name that is not a valid LLM tool name."""
    return "ok"


if __name__ == "__main__":
    server.run("stdio")
