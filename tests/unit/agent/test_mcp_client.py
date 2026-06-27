"""Tests for HTTP MCP client tool adaptation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from qmt_agent_trader.agent import mcp_client
from qmt_agent_trader.agent.mcp_client import (
    McpToolDescriptor,
    build_mcp_tools,
    load_mcp_server_config,
    sanitize_mcp_tool_name,
)
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


def test_load_mcp_config_expands_tavily_env_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    config_path = tmp_path / "mcp.servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "tavily",
                        "transport": "streamable_http",
                        "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}",
                        "headers": {"X-Test": "${TAVILY_API_KEY}"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config = load_mcp_server_config(config_path)

    assert config.servers[0].url == "https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-test-key"
    assert config.servers[0].headers == {"X-Test": "tvly-test-key"}


def test_sanitize_mcp_tool_name_is_openai_function_safe() -> None:
    assert (
        sanitize_mcp_tool_name(prefix="mcp", server_name="tavily", tool_name="tavily-search")
        == "mcp_tavily_tavily_search"
    )
    assert (
        sanitize_mcp_tool_name(prefix="mcp", server_name="news/api", tool_name="1.extract")
        == "mcp_news_api_1_extract"
    )


def test_build_mcp_tools_adapts_streamable_http_tool(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "mcp.servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "tavily",
                        "transport": "streamable_http",
                        "url": "https://mcp.tavily.com/mcp/",
                        "permission": "READ_ONLY",
                        "timeout_seconds": 45,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    listed_servers: list[str] = []
    called: list[tuple[str, str, dict[str, Any]]] = []

    def fake_list(server: mcp_client.McpServerConfig) -> list[McpToolDescriptor]:
        listed_servers.append(server.name)
        return [
            McpToolDescriptor(
                name="tavily-search",
                description="Search the web with Tavily.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]

    def fake_call(
        server: mcp_client.McpServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        called.append((server.name, tool_name, arguments))
        return {"structured": {"answer": "ok"}}

    monkeypatch.setattr(mcp_client, "_list_streamable_http_tools", fake_list)
    monkeypatch.setattr(mcp_client, "_call_streamable_http_tool", fake_call)

    tools = build_mcp_tools(
        config_path=config_path,
        tool_prefix="mcp",
        default_timeout_seconds=60,
    )

    assert listed_servers == ["tavily"]
    assert len(tools) == 1
    spec = tools[0].spec
    assert spec.name == "mcp_tavily_tavily_search"
    assert spec.description == "[MCP:tavily] Search the web with Tavily."
    assert spec.permission == PermissionLevel.READ_ONLY
    assert spec.timeout_seconds == 45
    assert spec.input_schema["required"] == ["query"]

    result = tools[0].run({"query": "QMT"}, ToolContext(run_id="run-1"))

    assert result == {"structured": {"answer": "ok"}}
    assert called == [("tavily", "tavily-search", {"query": "QMT"})]


def test_mcp_call_result_serialization_handles_text_structured_and_error() -> None:
    from mcp.types import CallToolResult, TextContent

    success = CallToolResult(
        content=[TextContent(type="text", text="hello")],
        structuredContent={"answer": "ok"},
        isError=False,
    )
    error = CallToolResult(
        content=[TextContent(type="text", text="failed")],
        isError=True,
    )

    assert mcp_client._serialize_call_tool_result(success) == {
        "is_error": False,
        "content": [{"type": "text", "text": "hello"}],
        "structured": {"answer": "ok"},
    }
    assert mcp_client._serialize_call_tool_result(error) == {
        "is_error": True,
        "content": [{"type": "text", "text": "failed"}],
        "error": True,
    }


@pytest.mark.anyio
async def test_run_async_works_inside_active_event_loop() -> None:
    async def sample() -> dict[str, str]:
        return {"status": "ok"}

    assert mcp_client._run_async(sample()) == {"status": "ok"}


def test_mcp_tool_permission_controls_llm_bridge(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "mcp.servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "tavily",
                        "transport": "streamable_http",
                        "url": "https://mcp.tavily.com/mcp/",
                        "permission": "APPROVAL_REQUIRED",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mcp_client,
        "_list_streamable_http_tools",
        lambda _server: [
            McpToolDescriptor(
                name="tavily-search",
                description="Search the web with Tavily.",
                input_schema={"type": "object", "properties": {}},
            )
        ],
    )

    registry = AgentToolRegistry()
    registry.register_all(
        *build_mcp_tools(
            config_path=config_path,
            tool_prefix="mcp",
            default_timeout_seconds=60,
        )
    )

    assert "mcp_tavily_tavily_search" in registry.tools
    assert "mcp_tavily_tavily_search" not in registry.to_legacy_registry().tools


def test_build_agent_registry_includes_enabled_http_mcp_tools(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "mcp.servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "tavily",
                        "transport": "streamable_http",
                        "url": "https://mcp.tavily.com/mcp/",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mcp_client,
        "_list_streamable_http_tools",
        lambda _server: [
            McpToolDescriptor(
                name="tavily-search",
                description="Search the web with Tavily.",
                input_schema={"type": "object", "properties": {}},
            )
        ],
    )

    registry = build_agent_registry(
        data_lake=DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "lake.duckdb"),
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        settings=Settings(
            project_root=tmp_path,
            mcp_enabled=True,
            mcp_config_path=config_path,
        ),
    )

    assert "mcp_tavily_tavily_search" in registry.tools
    assert "mcp_tavily_tavily_search" in registry.to_legacy_registry().tools
