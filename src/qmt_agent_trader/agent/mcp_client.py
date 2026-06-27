"""Streamable HTTP MCP client adapter for Agent tools."""

from __future__ import annotations

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

from loguru import logger
from pydantic import BaseModel, Field

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9]+")


class McpServerConfig(BaseModel):
    name: str
    transport: Literal["streamable_http"] = "streamable_http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    permission: PermissionLevel = PermissionLevel.READ_ONLY
    timeout_seconds: int | None = None
    sse_read_timeout_seconds: int | None = None
    llm_callable: bool = True


class McpServersConfig(BaseModel):
    servers: list[McpServerConfig] = Field(default_factory=list)


@dataclass(frozen=True)
class McpToolDescriptor:
    name: str
    description: str
    input_schema: dict[str, Any]


def load_mcp_server_config(path: Path) -> McpServersConfig:
    """Load MCP server config and expand ${ENV_VAR} values in URL/headers."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = McpServersConfig.model_validate(payload)
    return McpServersConfig(
        servers=[
            server.model_copy(
                update={
                    "url": _expand_env(server.url),
                    "headers": {
                        key: _expand_env(value) for key, value in server.headers.items()
                    },
                }
            )
            for server in config.servers
        ]
    )


def build_mcp_tools(
    *,
    config_path: Path,
    tool_prefix: str,
    default_timeout_seconds: int,
) -> list[AgentTool]:
    """Discover enabled MCP servers and adapt their tools into AgentTool objects."""
    if not config_path.exists():
        return []
    config = load_mcp_server_config(config_path)
    result: list[AgentTool] = []
    for server in config.servers:
        if not server.enabled:
            continue
        try:
            descriptors = _list_streamable_http_tools(server)
        except Exception as exc:
            logger.warning("Skipping MCP server '{}': {}", server.name, exc)
            continue
        for descriptor in descriptors:
            result.append(
                _build_mcp_tool(
                    server=server,
                    descriptor=descriptor,
                    tool_prefix=tool_prefix,
                    default_timeout_seconds=default_timeout_seconds,
                )
            )
    return result


def sanitize_mcp_tool_name(*, prefix: str, server_name: str, tool_name: str) -> str:
    parts = [_safe_name_part(prefix), _safe_name_part(server_name), _safe_name_part(tool_name)]
    name = "_".join(part for part in parts if part)
    if not name or not re.match(r"^[A-Za-z_]", name):
        name = f"mcp_{name}"
    return name[:64]


def _build_mcp_tool(
    *,
    server: McpServerConfig,
    descriptor: McpToolDescriptor,
    tool_prefix: str,
    default_timeout_seconds: int,
) -> AgentTool:
    tool_name = sanitize_mcp_tool_name(
        prefix=tool_prefix,
        server_name=server.name,
        tool_name=descriptor.name,
    )
    timeout = server.timeout_seconds or default_timeout_seconds

    def _run(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return _call_streamable_http_tool(server, descriptor.name, input_data)

    return tool(
        ToolSpec(
            name=tool_name,
            description=f"[MCP:{server.name}] {descriptor.description or descriptor.name}",
            input_schema=_object_schema(descriptor.input_schema),
            output_schema={"type": "object"},
            permission=server.permission,
            side_effect_level="none",
            timeout_seconds=timeout,
            deterministic=False,
            llm_callable=server.llm_callable,
        ),
        fn=_run,
    )


def _list_streamable_http_tools(server: McpServerConfig) -> list[McpToolDescriptor]:
    return cast(list[McpToolDescriptor], _run_async(_list_streamable_http_tools_async(server)))


async def _list_streamable_http_tools_async(server: McpServerConfig) -> list[McpToolDescriptor]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    timeout = float(server.timeout_seconds or 60)
    sse_timeout = float(server.sse_read_timeout_seconds or 300)
    async with streamablehttp_client(
        server.url,
        headers=server.headers or None,
        timeout=timeout,
        sse_read_timeout=sse_timeout,
    ) as (read_stream, write_stream, _session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [
                McpToolDescriptor(
                    name=item.name,
                    description=item.description or item.name,
                    input_schema=dict(item.inputSchema),
                )
                for item in tools.tools
            ]


def _call_streamable_http_tool(
    server: McpServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _run_async(_call_streamable_http_tool_async(server, tool_name, arguments)),
    )


async def _call_streamable_http_tool_async(
    server: McpServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    timeout = float(server.timeout_seconds or 60)
    sse_timeout = float(server.sse_read_timeout_seconds or 300)
    async with streamablehttp_client(
        server.url,
        headers=server.headers or None,
        timeout=timeout,
        sse_read_timeout=sse_timeout,
    ) as (read_stream, write_stream, _session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=timeout),
            )
            return _serialize_call_tool_result(result)


def _serialize_call_tool_result(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "is_error": bool(getattr(result, "isError", False)),
        "content": [
            _serialize_content_item(item)
            for item in list(getattr(result, "content", None) or [])
        ],
    }
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        payload["structured"] = _jsonable(structured)
    if payload["is_error"]:
        payload["error"] = True
    return payload


def _serialize_content_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return cast(
            dict[str, Any],
            item.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
    if isinstance(item, dict):
        return dict(item)
    return {"type": type(item).__name__, "value": str(item)}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)
    return value


def _object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "object":
        return schema
    return {
        "type": "object",
        "properties": dict(schema.get("properties", {})),
        "required": list(schema.get("required", [])),
        "additionalProperties": schema.get("additionalProperties", False),
    }


def _expand_env(value: str) -> str:
    return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)


def _safe_name_part(value: str) -> str:
    return _SAFE_NAME_PATTERN.sub("_", value).strip("_")


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()
