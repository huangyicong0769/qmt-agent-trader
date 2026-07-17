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


def _result(overlap: float | None) -> FactorRankResearchResult:
    return FactorRankResearchResult(
        metrics=SensitivityMetrics(total_return=0.0),
        trades=(),
        equity_points=(),
        rebalance_points=(),
        data_quality=ResearchDataQuality(),
        average_top_n_overlap=overlap,
    )


def _config() -> StrategyBacktestConfig:
    return StrategyBacktestConfig(
        strategy_id="factor_rank",
        strategy_identity_mode="adhoc",
        start_date="20240101",
        end_date="20240131",
    )


def test_no_comparable_selections_preserves_missing_overlap() -> None:
    metrics = execution_adapter._build_canonical_metrics(_result(None), _config())
    assert metrics["average_top_n_overlap"] is None


def test_computed_overlap_is_rounded_normally() -> None:
    metrics = execution_adapter._build_canonical_metrics(_result(0.1234567), _config())
    assert metrics["average_top_n_overlap"] == 0.123457


def test_missing_overlap_is_not_computed_in_diagnostics() -> None:
    result = _result(None)
    metrics = execution_adapter._build_canonical_metrics(result, _config())
    evidence = execution_adapter._diagnostic_evidence(
        result.as_dict(),
        {"valid": True},
        canonical_metrics=metrics,
        factor_frame=pd.DataFrame(),
        bars=pd.DataFrame(),
        initial_cash=1_000_000,
    )

    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence)
    checks = {check.name: check for check in diagnostics.checks}

    assert "average_top_n_overlap" not in evidence["churn_report"]
    assert checks["average_top_n_overlap"].status == DiagnosticStatus.NOT_COMPUTED
