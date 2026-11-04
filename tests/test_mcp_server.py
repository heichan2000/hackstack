"""The MCP server: who it answers as, and its two tools seen through a real MCP client.

Offline: an in-memory corpus (``tests/mcp_fixture.py``) behind the real server, reached through
the SDK's client both in-process and over stdio.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from hackstack.mcp_server import AS_USER_ENV, KBServer, build_server, resolve_user
from hackstack.retrieval.acl import UnknownUserError
from tests.mcp_fixture import APPS, REPO, build_kb, build_retriever

# --- who the server answers as ----------------------------------------------------------------


def test_user_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(AS_USER_ENV, "hr@northwind.test")
    kb = build_kb()
    assert kb.user == "hr@northwind.test"
    assert "group:hr" in kb.principals


def test_the_user_variable_is_not_mistaken_for_a_setting(monkeypatch):
    from hackstack.config import load_settings

    monkeypatch.setenv(AS_USER_ENV, "hr@northwind.test")  # settings reject unknown HACKSTACK_*
    assert load_settings(REPO / "hackstack.yaml", env_file=None).app == "company_brain"


def test_an_explicit_user_beats_the_environment(monkeypatch):
    monkeypatch.setenv(AS_USER_ENV, "hr@northwind.test")
    assert build_kb("eng@northwind.test").user == "eng@northwind.test"


def test_without_a_user_it_falls_back_to_the_apps_default_user():
    kb = build_kb()
    assert kb.user == "eng@northwind.test"  # company_brain's default_user
    assert kb.user_source == "app default_user"


def test_an_unknown_user_fails_at_startup_instead_of_searching_as_nobody():
    with pytest.raises(UnknownUserError):
        build_kb("stranger@elsewhere.test")


def test_no_user_and_no_default_user_is_an_error(tmp_path):
    from hackstack.config import Settings

    with pytest.raises(UnknownUserError):
        resolve_user(Settings(app="no_such_app"), None, apps_dir=tmp_path)


def test_the_retriever_is_built_lazily_on_first_use():
    built = []

    def factory():
        built.append(1)
        return build_retriever()

    from hackstack.config import load_settings

    settings = load_settings(REPO / "hackstack.yaml", profile="default", env_file=None)
    kb = KBServer(settings, user="eng@northwind.test", apps_dir=APPS, retriever_factory=factory)
    assert built == []
    kb.search("PTO carryover")
    kb.search("deploys")
    assert built == [1]


def test_warm_up_builds_the_retriever_and_loads_the_reranker_on_the_calling_thread():
    import threading

    loaded = []
    retriever = build_retriever()
    retriever.reranker.load = lambda: loaded.append(threading.current_thread())

    from hackstack.config import load_settings

    settings = load_settings(REPO / "hackstack.yaml", profile="default", env_file=None)
    kb = KBServer(
        settings, user="eng@northwind.test", apps_dir=APPS, retriever_factory=lambda: retriever
    )
    kb.warm_up()
    assert loaded == [threading.current_thread()]


# --- the tools through an MCP client -----------------------------------------------------------


@asynccontextmanager
async def client(user: str):
    async with create_connected_server_and_client_session(build_server(build_kb(user))) as s:
        yield s


def text_of(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def call(user: str, tool: str, args: dict):
    async def go():
        async with client(user) as s:
            return await s.call_tool(tool, args)

    return anyio.run(go)


def test_lists_search_kb_and_get_document_with_their_schemas():
    async def go():
        async with client("eng@northwind.test") as s:
            return (await s.list_tools()).tools

    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == {"search_kb", "get_document"}
    assert tools["search_kb"].inputSchema["required"] == ["query"]
    assert set(tools["search_kb"].inputSchema["properties"]) == {"query", "source", "since"}
    assert tools["get_document"].inputSchema["required"] == ["document"]
    assert "knowledge base" in tools["search_kb"].description
    assert tools["search_kb"].annotations.readOnlyHint is True
    # the identity is fixed by the server's config, never a tool argument the model could set
    assert all("user" not in t.inputSchema["properties"] for t in tools.values())


def test_search_returns_numbered_passages_with_titles_and_source_ids():
    result = call("eng@northwind.test", "search_kb", {"query": "PTO carryover days"})
    assert not result.isError
    text = text_of(result)
    assert "[1] PTO Policy" in text
    assert "kb://" in text
    assert "five unused PTO days" in text


def test_search_respects_the_acl():
    eng = text_of(call("eng@northwind.test", "search_kb", {"query": "salary band E4"}))
    hr = text_of(call("hr@northwind.test", "search_kb", {"query": "salary band E4"}))
    assert "165k" not in eng and "Compensation Bands" not in eng
    assert "165k" in hr


def test_search_with_no_visible_match_says_so():
    result = call("finance@northwind.test", "search_kb", {"query": "Friday deploy freeze"})
    assert "Deploy Runbook" not in text_of(result)


def test_a_bad_since_is_a_tool_error_the_model_can_fix():
    result = call("eng@northwind.test", "search_kb", {"query": "PTO", "since": "last week"})
    assert result.isError
    assert "since" in text_of(result)


def test_get_document_by_title_after_a_search_and_by_source_id():
    async def go():
        async with client("eng@northwind.test") as s:
            await s.call_tool("search_kb", {"query": "PTO carryover"})
            by_title = await s.call_tool("get_document", {"document": "PTO Policy"})
            by_id = await s.call_tool("get_document", {"document": "handbook/pto.md"})
            return by_title, by_id

    by_title, by_id = anyio.run(go)
    for result in (by_title, by_id):
        text = text_of(result)
        assert "five unused PTO days" in text and "two weeks ahead" in text


def test_get_document_never_returns_a_restricted_document():
    result = call("eng@northwind.test", "get_document", {"document": "hr/comp-bands.md"})
    text = text_of(result)
    assert "165k" not in text
    assert "No document matching" in text


# --- over stdio, as Claude Desktop runs it -----------------------------------------------------


def test_lists_and_calls_both_tools_over_stdio():
    params = StdioServerParameters(
        command=sys.executable, args=[str(REPO / "tests" / "mcp_fixture.py"), "hr@northwind.test"]
    )

    async def go():
        with anyio.fail_after(60):
            return await session()

    async def session():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
            init = await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            search = await s.call_tool("search_kb", {"query": "salary band E4"})
            doc = await s.call_tool("get_document", {"document": "Compensation Bands"})
            return init, tools, search, doc

    init, tools, search, doc = anyio.run(go)
    assert init.serverInfo.name == "hackstack"
    assert sorted(tools) == ["get_document", "search_kb"]
    assert "165k" in text_of(search)
    assert "165k" in text_of(doc)
