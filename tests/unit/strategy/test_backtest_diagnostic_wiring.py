from __future__ import annotations

from dataclasses import replace

import pandas as pd

from qmt_agent_trader.backtest.research_models import (
    FactorRankResearchResult,
    ResearchDataQuality,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityMetrics
from qmt_agent_trader.strategy import execution_adapter
from qmt_agent_trader.strategy.diagnostics import (
    DiagnosticStatus,
    StrategyDiagnosticsEvaluator,
)
from qmt_agent_trader.strategy.execution_adapter import StrategyBacktestConfig


def factor_rank_result() -> FactorRankResearchResult:
    return FactorRankResearchResult(
        metrics=SensitivityMetrics(total_return=-0.20),
        trades=(),
        equity_points=(),
        rebalance_points=(),
        data_quality=ResearchDataQuality(),
        total_explicit_cost=10_000.0,
        total_slippage_cost=5_000.0,
        same_trade_gross_return=0.05,
        average_top_n_overlap=0.80,
    )


def minimal_evidence() -> dict[str, object]:
    return {
        "leakage_report": {"valid": True},
        "factor_report": {"observation_count": 300},
        "performance_report": {"max_drawdown": 0.02},
        "turnover_report": {"average_turnover": 0.1},
        "cost_report": {"cost_to_initial_cash": 0.001},
        "rejection_report": {"rate": 0.0},
        "trade_blotter": {"count": 30},
    }


def test_diagnostics_use_canonical_cost_drag_and_overlap() -> None:
    result = factor_rank_result()
    config = StrategyBacktestConfig(
        strategy_id="factor_rank",
        strategy_identity_mode="adhoc",
        start_date="20240101",
        end_date="20240331",
        initial_cash=1_000_000,
    )
    factor_frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2024-01-02")],
            "factor_value": [1.0],
        }
    )
    bars = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2024-01-02")],
            "close": [10.0],
        }
    )

    metrics = execution_adapter._build_canonical_metrics(result, config)
    evidence = execution_adapter._diagnostic_evidence(
        result.as_dict(),
        {"valid": True, "execution_delay_days": 1},
        canonical_metrics=metrics,
        factor_frame=factor_frame,
        bars=bars,
        initial_cash=config.initial_cash,
    )
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence)

    checks = {check.name: check for check in diagnostics.checks}
    assert checks["cost_drag"].observed == 0.25
    assert checks["average_top_n_overlap"].observed == 0.80


def test_missing_cost_drag_is_not_computed_not_passed() -> None:
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(minimal_evidence())
    checks = {check.name: check for check in diagnostics.checks}
    assert checks["cost_drag"].status == DiagnosticStatus.NOT_COMPUTED


def test_missing_overlap_is_not_computed_not_passed() -> None:
    result = replace(factor_rank_result(), average_top_n_overlap=None)
    config = StrategyBacktestConfig(
        strategy_id="factor_rank",
        strategy_identity_mode="adhoc",
        start_date="20240101",
        end_date="20240331",
    )
    metrics = execution_adapter._build_canonical_metrics(result, config)
    evidence = execution_adapter._diagnostic_evidence(
        result.as_dict(),
        {"valid": True},
        canonical_metrics=metrics,
        factor_frame=pd.DataFrame(),
        bars=pd.DataFrame(),
        initial_cash=config.initial_cash,
    )
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence)
    checks = {check.name: check for check in diagnostics.checks}
    assert checks["average_top_n_overlap"].status == DiagnosticStatus.NOT_COMPUTED
