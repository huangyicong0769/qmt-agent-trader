from __future__ import annotations

import time

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools.base import tool


def test_tool_process_timeout_does_not_deadlock_on_large_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    registry = AgentToolRegistry()
    registry.register(
        tool(
            ToolSpec(
                name="query_bars",
                description="large payload",
                permission=PermissionLevel.READ_ONLY,
                timeout_seconds=5,
            ),
            fn=lambda _data, _ctx: {"blob": "x" * (5 * 1024 * 1024)},
        )
    )

    started = time.monotonic()
    result = registry.run_tool("query_bars", {}, ToolContext(run_id="large-payload"))

    assert time.monotonic() - started < 5
    assert result["blob"].startswith("xxx")
    assert result["payload_file"].endswith("large-payload_query_bars.json")
    assert result["payload_size_bytes"] > 1_000_000


def test_tool_process_timeout_returns_structured_timeout() -> None:
    registry = AgentToolRegistry()
    registry.register(
        tool(
            ToolSpec(
                name="query_bars",
                description="slow",
                permission=PermissionLevel.READ_ONLY,
                timeout_seconds=0,
            ),
            fn=lambda _data, _ctx: time.sleep(0.2) or {"late": True},
        )
    )

    result = registry.run_tool("query_bars", {}, ToolContext(run_id="slow-process"))

    assert result["status"] == "TIMEOUT"
    assert result["tool_name"] == "query_bars"
    assert result["kill_attempted"] is True
    assert result["partial_payload"] is False
