"""Backtest report helpers."""

from __future__ import annotations

from qmt_agent_trader.backtest.engine import BacktestResult


def summarize_result(result: BacktestResult) -> dict[str, object]:
    return {"fills": len(result.fills), "leakage_valid": result.leakage_valid}
