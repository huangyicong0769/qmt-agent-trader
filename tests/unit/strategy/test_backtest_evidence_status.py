from __future__ import annotations

from qmt_agent_trader.agent.tools.strategy_tools import _with_backtest_evidence_status


def test_backtest_completed_with_failed_diagnostics_exposes_invalid_evidence() -> None:
    payload = {
        "status": "completed",
        "diagnostics": {"status": "FAIL"},
        "metrics": {"max_drawdown": -0.99},
    }

    result = _with_backtest_evidence_status(payload)

    assert result["status"] == "completed"
    assert result["raw_status"] == "completed"
    assert result["diagnostic_status"] == "FAIL"
    assert result["domain_status"] == "FAILED"
    assert result["evidence_status"] == "INVALID"
    assert result["recommendation_status"] == "DO_NOT_RECOMMEND"
