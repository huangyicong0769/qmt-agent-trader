"""Agent backtest tools."""

from __future__ import annotations

from pathlib import Path

from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityAnalyzer, SensitivityGrid
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.services.research_report_service import save_research_report


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
        raise ValueError(
            "no daily bars found in data lake; run data fetch and build_data_table first"
        )
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


def run_factor_rank_sensitivity_report(
    lake: DataLake,
    reports_dir: Path,
    *,
    factor_name: str,
    cost_multipliers: list[float] | None = None,
    slippage_bps: list[float] | None = None,
    execution_delay_days: list[int] | None = None,
    top_n: list[int] | None = None,
    max_single_position_pct: list[float] | None = None,
    initial_cash: float = 1_000_000.0,
    agent_notes: str | None = None,
    infrastructure_requests: list[str] | None = None,
) -> dict[str, object]:
    """Run sensitivity analysis and persist it as reviewable research evidence."""
    result = run_factor_rank_sensitivity(
        lake,
        factor_name=factor_name,
        cost_multipliers=cost_multipliers,
        slippage_bps=slippage_bps,
        execution_delay_days=execution_delay_days,
        top_n=top_n,
        max_single_position_pct=max_single_position_pct,
        initial_cash=initial_cash,
    )
    return save_research_report(
        reports_dir,
        artifact_type="factor_rank_sensitivity",
        title=f"Factor-rank sensitivity: {factor_name}",
        payload=result,
        metadata={
            "factor_name": factor_name,
            "initial_cash": initial_cash,
            "grid": _sensitivity_grid_metadata(
                cost_multipliers=cost_multipliers,
                slippage_bps=slippage_bps,
                execution_delay_days=execution_delay_days,
                top_n=top_n,
                max_single_position_pct=max_single_position_pct,
            ),
        },
        agent_notes=agent_notes,
        infrastructure_requests=infrastructure_requests,
    )


def _sensitivity_grid_metadata(
    *,
    cost_multipliers: list[float] | None,
    slippage_bps: list[float] | None,
    execution_delay_days: list[int] | None,
    top_n: list[int] | None,
    max_single_position_pct: list[float] | None,
) -> dict[str, object]:
    return {
        "cost_multipliers": cost_multipliers or [1.0, 2.0, 3.0],
        "slippage_bps": slippage_bps or [0.0, 5.0, 10.0],
        "execution_delay_days": execution_delay_days or [1, 2],
        "top_n": top_n or [10, 20],
        "max_single_position_pct": max_single_position_pct or [0.05, 0.10],
    }
