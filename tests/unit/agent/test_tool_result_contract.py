from __future__ import annotations

from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools.base import tool


def test_blocked_tool_result_preserves_raw_payload_and_exposes_status() -> None:
    registry = AgentToolRegistry()
    registry.register(
        tool(
            ToolSpec(name="blocked_tool", description="blocked"),
            fn=lambda _data, _ctx: {
                "status": "BLOCKED",
                "reason": "MISSING_FACTOR_INPUTS",
            },
        )
    )

    result = registry.run_tool("blocked_tool", {}, ToolContext(run_id="r-blocked"))

    assert result["execution_status"] == "OK"
    assert result["domain_status"] == "BLOCKED"
    assert result["evidence_status"] == "BLOCKED"
    assert result["recommendation_status"] == "BLOCKED"
    assert result["raw_status"] == "BLOCKED"
    assert result["result"] == {
        "status": "BLOCKED",
        "reason": "MISSING_FACTOR_INPUTS",
    }


def test_legacy_unstructured_result_is_unknown_not_success() -> None:
    registry = AgentToolRegistry()
    registry.register(
        tool(
            ToolSpec(name="legacy_tool", description="legacy"),
            fn=lambda _data, _ctx: {"rows": [{"x": 1}]},
        )
    )

    result = registry.run_tool("legacy_tool", {}, ToolContext(run_id="r-legacy"))

    assert result["execution_status"] == "OK"
    assert result["domain_status"] == "UNKNOWN"
    assert result["evidence_status"] == "UNKNOWN"
    assert result["recommendation_status"] == "UNKNOWN"
    assert "legacy_unstructured_tool_result" in result["warnings"]


def test_completed_backtest_with_failed_diagnostics_is_invalid_evidence() -> None:
    registry = AgentToolRegistry()
    registry.register(
        tool(
            ToolSpec(name="backtest_like", description="backtest"),
            fn=lambda _data, _ctx: {
                "status": "completed",
                "diagnostics": {"status": "FAIL"},
            },
        )
    )

    result = registry.run_tool("backtest_like", {}, ToolContext(run_id="r-backtest"))

    assert result["raw_status"] == "completed"
    assert result["diagnostic_status"] == "FAIL"
    assert result["domain_status"] == "FAILED"
    assert result["evidence_status"] == "INVALID"
    assert result["recommendation_status"] == "DO_NOT_RECOMMEND"
