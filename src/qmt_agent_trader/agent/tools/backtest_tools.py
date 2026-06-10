"""Agent backtest tools."""

from __future__ import annotations

from qmt_agent_trader.agent.permissions import ToolCapability

CAPABILITY = ToolCapability.RUN_BACKTEST


def run_simulated_backtest(strategy_id: str) -> dict[str, object]:
    return {"strategy_id": strategy_id, "mode": "simulation", "valid": True}
