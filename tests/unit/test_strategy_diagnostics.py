from qmt_agent_trader.strategy.diagnostics import (
    DiagnosticStatus,
    StrategyDiagnosticConfig,
    StrategyDiagnosticsEvaluator,
)


def test_strategy_diagnostics_warn_on_thin_evidence() -> None:
    evidence = {
        "leakage_report": {"valid": True},
        "factor_report": {
            "observation_count": 4,
            "coverage": 1.0,
            "positive_ic_ratio": 1.0,
            "walk_forward": [
                {"mean_ic": 0.1, "long_short_spread": 0.02},
                {"mean_ic": 0.2, "long_short_spread": 0.03},
            ],
        },
        "performance_report": {"max_drawdown": 0.0},
        "turnover_report": {"turnovers": [0.2, 0.3]},
        "cost_report": {"cost_to_initial_cash": 0.001},
        "rejection_report": {"rate": 0.0},
        "trade_blotter": {"count": 1},
    }

    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence)

    assert diagnostics.status == DiagnosticStatus.WARN
    warning_names = {
        check.name for check in diagnostics.checks if check.status == DiagnosticStatus.WARN
    }
    assert warning_names == {"min_observations", "min_trade_count"}


def test_strategy_diagnostics_fail_on_invalid_leakage_or_large_drawdown() -> None:
    evidence = {
        "leakage_report": {"valid": False},
        "factor_report": {"observation_count": 300},
        "performance_report": {"max_drawdown": -0.4},
        "trade_blotter": {"count": 30},
    }

    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence)

    assert diagnostics.status == DiagnosticStatus.FAIL
    failed_names = {
        check.name for check in diagnostics.checks if check.status == DiagnosticStatus.FAIL
    }
    assert failed_names == {"leakage_valid", "max_drawdown"}


def test_strategy_diagnostics_pass_when_thresholds_are_met() -> None:
    evidence = {
        "leakage_report": {"valid": True},
        "factor_report": {
            "observation_count": 4,
            "coverage": 1.0,
            "positive_ic_ratio": 1.0,
            "walk_forward": [{"mean_ic": 0.1, "long_short_spread": 0.02}],
        },
        "performance_report": {"max_drawdown": 0.02},
        "turnover_report": {"average_turnover": 0.1},
        "cost_report": {"cost_to_initial_cash": 0.001},
        "rejection_report": {"rate": 0.0},
        "trade_blotter": {"count": 1},
    }
    config = StrategyDiagnosticConfig(min_observations=1, min_trade_count=1)

    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence, config)

    assert diagnostics.status == DiagnosticStatus.PASS


def test_strategy_diagnostics_marks_missing_factor_evidence_not_computed() -> None:
    evidence = {
        "leakage_report": {"valid": True},
        "factor_report": {"observation_count": 300},
        "performance_report": {"max_drawdown": 0.02},
        "turnover_report": {"average_turnover": 0.1},
        "cost_report": {"cost_to_initial_cash": 0.001},
        "rejection_report": {"rate": 0.0},
        "trade_blotter": {"count": 30},
    }

    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence)
    payload = diagnostics.as_dict()
    checks = {check["name"]: check for check in payload["checks"]}

    assert diagnostics.status == DiagnosticStatus.WARN
    assert checks["coverage"]["status"] == "NOT_COMPUTED"
    assert checks["coverage"]["evidence_source"] == "not_computed"
    assert checks["positive_ic_ratio"]["status"] == "NOT_COMPUTED"
    assert checks["positive_ic_ratio"]["evidence_source"] == "not_computed"
    assert checks["walk_forward_consistency"]["status"] == "NOT_COMPUTED"
    assert checks["walk_forward_consistency"]["evidence_source"] == "not_computed"
