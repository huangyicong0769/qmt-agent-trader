from __future__ import annotations

import pytest

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.backtest.errors import BacktestAccountingError
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def wired_strategy_tools(tmp_path, monkeypatch) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "test.duckdb")
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)


def valid_backtest_input() -> dict[str, object]:
    return {
        "factor_name": "momentum_20d",
        "symbols": ["000001.SZ"],
        "start_date": "20240102",
        "end_date": "20240102",
    }


def tool_context() -> ToolContext:
    return ToolContext(run_id="integrity-error-boundary")


def test_agent_tool_returns_structured_error_for_known_integrity_failure(
    monkeypatch,
    wired_strategy_tools,
) -> None:
    def fail(*_args, **_kwargs):
        raise BacktestAccountingError(
            code="NEGATIVE_CASH_AFTER_BUY",
            message="cash became negative",
            trade_date="2024-01-03",
            symbols=("000001.SZ",),
            field="cash",
            details={"cash": -5.0},
        )

    monkeypatch.setattr(strategy_tools, "run_strategy_backtest", fail)

    result = strategy_tools._run_backtest(valid_backtest_input(), tool_context())

    assert result["status"] == "ERROR"
    assert result["reason"] == "BACKTEST_INTEGRITY_ERROR"
    assert result["error"]["code"] == "NEGATIVE_CASH_AFTER_BUY"
    assert result["error"]["details"]["cash"] == -5.0


def test_agent_tool_does_not_swallow_unexpected_runtime_error(
    monkeypatch,
    wired_strategy_tools,
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(strategy_tools, "run_strategy_backtest", fail)

    with pytest.raises(RuntimeError, match="bug"):
        strategy_tools._run_backtest(valid_backtest_input(), tool_context())
