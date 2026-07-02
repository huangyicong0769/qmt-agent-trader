from __future__ import annotations

from qmt_agent_trader.agent.llm_client import ToolResult
from qmt_agent_trader.agent.orchestrator import _done_message, _stream_to_events


def test_tool_done_event_displays_failed_evidence_not_checkmark() -> None:
    events = _stream_to_events(
        ToolResult(
            tool_call_id="call-1",
            tool_name="run_backtest",
            result={
                "execution_status": "OK",
                "domain_status": "FAILED",
                "evidence_status": "INVALID",
                "recommendation_status": "DO_NOT_RECOMMEND",
                "raw_status": "completed",
                "diagnostic_status": "FAIL",
            },
        ),
        "run-ui",
        "exp-ui",
    )

    assert events[0].type == "tool_done"
    assert events[0].message == "Tool: run_backtest ✗"
    assert events[0].data["domain_status"] == "FAILED"
    assert events[0].data["evidence_status"] == "INVALID"


def test_tool_done_event_displays_unknown_for_legacy_result() -> None:
    events = _stream_to_events(
        ToolResult(
            tool_call_id="call-1",
            tool_name="legacy_tool",
            result={
                "execution_status": "OK",
                "domain_status": "UNKNOWN",
                "evidence_status": "UNKNOWN",
                "recommendation_status": "UNKNOWN",
            },
        ),
        "run-ui",
        "exp-ui",
    )

    assert events[0].message == "Tool: legacy_tool ?"


def test_done_message_reports_invalid_evidence_not_success() -> None:
    message = _done_message(
        {
            "summary": {
                "valid_count": 0,
                "weak_count": 0,
                "invalid_count": 1,
                "blocked_count": 0,
                "incomplete_count": 0,
                "unknown_count": 0,
            }
        }
    )

    assert message == "Run finished: all tool calls returned, evidence invalid."
