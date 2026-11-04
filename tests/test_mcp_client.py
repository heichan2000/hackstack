"""The MCP client adapter against a tiny local stdio server (tests/fixtures/mcp_fixture_server)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from hackstack.agent.loop import collect, stream_agent  # noqa: E402
from hackstack.agent.tools import Passages, ToolContext, ToolInputError  # noqa: E402
from hackstack.config import MCPServerSettings, load_settings  # noqa: E402
from hackstack.mcp_client import MCPClient, expand_env, prefixed_name  # noqa: E402
from hackstack.models import TextDelta, ToolCall  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_fixture_server.py"


def fixture_server(**overrides) -> MCPServerSettings:
    return MCPServerSettings(command=sys.executable, args=[str(FIXTURE)], **overrides)


@pytest.fixture(scope="module")
def client():
    c = MCPClient({"fx": fixture_server()})
    c.start()
    yield c
    c.close()


def tools_by_name(client: MCPClient):
    return {t.name: t for t in client.tools}


def call(tool, args=None):
    return asyncio.run(tool.handler(args or {}, ToolContext(principals=["public"])))


# --- naming ------------------------------------------------------------------------------------


def test_prefixed_name_joins_server_and_tool():
    assert prefixed_name("github", "search_issues") == "github__search_issues"


def test_prefixed_name_replaces_characters_llm_apis_refuse():
    assert prefixed_name("fx", "weird.name/with spaces") == "fx__weird_name_with_spaces"


def test_prefixed_name_stays_within_64_chars_and_unique():
    a = prefixed_name("server", "x" * 100)
    b = prefixed_name("server", "x" * 99 + "y")
    assert len(a) <= 64 and len(b) <= 64
    assert a != b


# --- config ------------------------------------------------------------------------------------


def test_mcp_servers_load_from_yaml(tmp_path):
    cfg = tmp_path / "hackstack.yaml"
    cfg.write_text(
        "mcp_servers:\n"
        "  github:\n"
        "    command: npx\n"
        "    args: ['-y', '@modelcontextprotocol/server-github']\n"
        "    env: {GITHUB_PERSONAL_ACCESS_TOKEN: '${GITHUB_TOKEN}'}\n"
        "    tools: [search_issues]\n",
        encoding="utf-8",
    )
    s = load_settings(cfg, env_file=None)
    gh = s.mcp_servers["github"]
    assert gh.command == "npx"
    assert gh.args == ["-y", "@modelcontextprotocol/server-github"]
    assert gh.tools == ["search_issues"]


def test_mcp_server_settings_refuse_typos():
    with pytest.raises(ValueError):
        MCPServerSettings(command="x", argz=[])


def test_expand_env_fills_placeholders_from_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    assert expand_env({"T": "${GITHUB_TOKEN}", "X": "plain"}) == {"T": "ghp_secret", "X": "plain"}


def test_expand_env_drops_unset_placeholders(monkeypatch):
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    assert expand_env({"T": "${NOPE_NOT_SET}"}) == {}


# --- listing -----------------------------------------------------------------------------------


def test_tools_are_listed_under_prefixed_names(client):
    names = set(tools_by_name(client))
    assert {"fx__echo", "fx__two_notes", "fx__readme", "fx__chart", "fx__broken"} <= names
    assert "fx__weird_name_with_spaces" in names


def test_listed_tools_keep_schema_and_say_where_they_come_from(client):
    echo = tools_by_name(client)["fx__echo"]
    assert echo.input_schema["type"] == "object"
    assert "text" in echo.input_schema["properties"]
    assert "Echo the text back." in echo.description
    assert "MCP server 'fx'" in echo.description


def test_allowlist_limits_registered_tools():
    c = MCPClient({"fx": fixture_server(tools=["echo"])})
    try:
        assert [t.name for t in c.start()] == ["fx__echo"]
    finally:
        c.close()


def test_disabled_server_is_not_started():
    c = MCPClient({"fx": fixture_server(enabled=False)})
    try:
        assert c.start() == []
    finally:
        c.close()


def test_server_that_cannot_start_is_skipped_with_a_warning(caplog):
    bad = MCPServerSettings(command=sys.executable, args=["-c", "import sys; sys.exit(3)"])
    c = MCPClient({"bad": bad, "fx": fixture_server(tools=["echo"])}, startup_timeout=15)
    try:
        tools = c.start()
    finally:
        c.close()
    assert [t.name for t in tools] == ["fx__echo"]
    assert any("bad" in r.getMessage() for r in caplog.records)


# --- calling -----------------------------------------------------------------------------------


def test_text_output_is_wrapped_as_citable_passages(client):
    out = call(tools_by_name(client)["fx__two_notes"])
    assert isinstance(out, Passages)
    assert [h.text for h in out.hits] == [
        "First note: the deploy freeze starts Friday.",
        "Second note: hotfixes need two approvals.",
    ]
    assert all(h.source == "mcp:fx" for h in out.hits)
    assert len({h.chunk_id for h in out.hits}) == 2
    assert out.skip_seen is False


def test_each_call_gets_fresh_source_ids(client):
    echo = tools_by_name(client)["fx__echo"]
    a = call(echo, {"text": "hi"})
    b = call(echo, {"text": "hi"})
    assert a.hits[0].chunk_id != b.hits[0].chunk_id


def test_embedded_text_resource_is_wrapped_with_its_uri(client):
    out = call(tools_by_name(client)["fx__readme"])
    assert isinstance(out, Passages)
    assert out.hits[0].text == "# Repo\nHello."
    assert out.hits[0].url == "file:///repo/README.md"


def test_output_with_an_image_is_not_wrapped(client):
    out = call(tools_by_name(client)["fx__chart"])
    assert isinstance(out, list)
    assert [b["type"] for b in out] == ["text", "image"]
    assert out[1]["source"]["media_type"] == "image/png"
    assert not any(b["type"] == "search_result" for b in out)


def test_tool_error_becomes_a_tool_input_error(client):
    with pytest.raises(ToolInputError, match="backend is down"):
        call(tools_by_name(client)["fx__broken"])


def test_tools_are_callable_from_any_event_loop(client):
    echo = tools_by_name(client)["fx__echo"]

    async def two_parallel():
        ctx = ToolContext(principals=["public"])
        return await asyncio.gather(
            echo.handler({"text": "a"}, ctx), echo.handler({"text": "b"}, ctx)
        )

    a, b = asyncio.run(two_parallel())
    assert a.hits[0].text == "echo: a" and b.hits[0].text == "echo: b"


def test_calling_after_close_is_a_clear_error():
    c = MCPClient({"fx": fixture_server(tools=["echo"])})
    (echo,) = c.start()
    c.close()
    with pytest.raises(RuntimeError, match="not connected"):
        call(echo, {"text": "x"})


# --- through the agent loop --------------------------------------------------------------------


def test_agent_loop_calls_mcp_tools_and_cites_their_output(client):
    llm = FakeLLM(
        [
            [
                ToolCall(id="c1", name="fx__echo", input={"text": "ping"}),
                ToolCall(id="c2", name="fx__chart", input={}),
            ],
            [TextDelta(text="It said ping [1].")],
        ]
    )
    result = asyncio.run(
        collect(
            stream_agent(
                llm,
                system="sys",
                messages=[{"role": "user", "content": "say ping"}],
                tools=client.tools,
                ctx=ToolContext(principals=["public"]),
            )
        )
    )
    assert result.answer == "It said ping [1]."
    assert [t["name"] for t in llm.calls[0]["tools"]] == [t.name for t in client.tools]
    results = llm.calls[1]["messages"][-1]["content"]
    echo_result, chart_result = results
    assert echo_result["content"][0]["type"] == "search_result"
    assert echo_result["content"][0]["source"].startswith("kb://mcp:fx/echo")
    assert [b["type"] for b in chart_result["content"]] == ["text", "image"]
    assert result.sources[0].title.startswith("fx")


def test_agent_from_settings_adds_mcp_tools_after_its_own(tmp_path):
    from hackstack.agent.runtime import Agent
    from hackstack.config import Settings

    settings = Settings(
        _env_file=None, mcp_servers={"fx": fixture_server(tools=["echo"]).model_dump()}
    )
    agent = Agent.from_settings(settings, apps_dir=tmp_path, llm=FakeLLM(), retriever=object())
    try:
        assert [t.name for t in agent.tools] == ["search_kb", "get_document", "fx__echo"]
    finally:
        agent.close()
    agent.close()  # twice is fine


def test_agent_without_mcp_servers_starts_none(tmp_path):
    from hackstack.agent.runtime import Agent
    from hackstack.config import Settings

    agent = Agent.from_settings(
        Settings(_env_file=None), apps_dir=tmp_path, llm=FakeLLM(), retriever=object()
    )
    assert agent.mcp is None
    agent.close()
