"""Agent backtest tools."""

from __future__ import annotations

from qmt_agent_trader.agent.permissions import ToolCapability
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityAnalyzer, SensitivityGrid
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake

CAPABILITY = ToolCapability.RUN_BACKTEST


def run_simulated_backtest(strategy_id: str) -> dict[str, object]:
    return {"strategy_id": strategy_id, "mode": "simulation", "valid": True}


def plan_sensitivity_analysis(
    cost_multipliers: list[float] | None = None,
    slippage_bps: list[float] | None = None,
    execution_delay_days: list[int] | None = None,
    top_n: list[int] | None = None,
    max_single_position_pct: list[float] | None = None,
) -> dict[str, object]:
    """Build a robustness scenario grid without pretending to run it."""
    grid = SensitivityGrid(
        cost_multipliers=tuple(cost_multipliers or [1.0, 2.0, 3.0]),
        slippage_bps=tuple(slippage_bps or [0.0, 5.0, 10.0]),
        execution_delay_days=tuple(execution_delay_days or [1, 2]),
        top_n=tuple(top_n or [10, 20]),
        max_single_position_pct=tuple(max_single_position_pct or [0.05, 0.10]),
    )
    scenarios = grid.scenarios()
    return {
        "status": "planned",
        "scenario_count": len(scenarios),
        "scenarios": [
            {"label": scenario.label(), **scenario.__dict__}
            for scenario in scenarios
        ],
        "runner_contract": {
            "required_metrics": [
                "total_return",
                "sharpe",
                "max_drawdown",
                "turnover",
                "diagnostic_pass",
            ],
            "notes": (
                "A strategy-specific runner must execute each scenario with the requested "
                "cost multiplier, slippage, execution delay, top_n, and max position cap. "
                "If any dimension cannot be executed, the agent should report that missing "
                "runner capability before comparing strategies."
            ),
        },
    }


def run_factor_rank_sensitivity(
    lake: DataLake,
    factor_name: str,
    cost_multipliers: list[float] | None = None,
    slippage_bps: list[float] | None = None,
    execution_delay_days: list[int] | None = None,
    top_n: list[int] | None = None,
    max_single_position_pct: list[float] | None = None,
    initial_cash: float = 1_000_000.0,
) -> dict[str, object]:
    """Run a real data-lake factor-rank robustness simulation."""
    bars = load_daily_bars(lake)
    if bars.empty:
        raise ValueError("no daily bars found in data lake; run data update first")
    grid = SensitivityGrid(
        cost_multipliers=tuple(cost_multipliers or [1.0, 2.0, 3.0]),
        slippage_bps=tuple(slippage_bps or [0.0, 5.0, 10.0]),
        execution_delay_days=tuple(execution_delay_days or [1, 2]),
        top_n=tuple(top_n or [10, 20]),
        max_single_position_pct=tuple(max_single_position_pct or [0.05, 0.10]),
    )
    runner = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name=factor_name,
            top_n=next(value for value in grid.top_n if value is not None),
            max_single_position_pct=next(
                value for value in grid.max_single_position_pct if value is not None
            ),
            initial_cash=initial_cash,
        ),
    )
    report = SensitivityAnalyzer().run(
        grid.scenarios(),
        lambda scenario: runner.run(scenario).metrics,
    )
    return {"status": "completed", **report.as_dict()}
