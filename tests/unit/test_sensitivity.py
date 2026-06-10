import pytest

from qmt_agent_trader.backtest.sensitivity import (
    SensitivityAnalyzer,
    SensitivityGrid,
    SensitivityMetrics,
    SensitivityScenario,
)


def test_sensitivity_grid_expands_cartesian_product() -> None:
    grid = SensitivityGrid(
        cost_multipliers=(1.0, 2.0),
        slippage_bps=(0.0, 5.0),
        execution_delay_days=(1, 2),
        top_n=(10, 20),
        max_single_position_pct=(0.05,),
    )

    scenarios = grid.scenarios()

    assert len(scenarios) == 16
    assert scenarios[0].label() == "cost_x1__slip_0bps__delay_1d__top_10__maxpos_0.05"


def test_sensitivity_analyzer_summarizes_worst_case_and_pass_ratio() -> None:
    scenarios = [
        SensitivityScenario(cost_multiplier=1.0, slippage_bps=0.0, execution_delay_days=1),
        SensitivityScenario(cost_multiplier=2.0, slippage_bps=5.0, execution_delay_days=1),
        SensitivityScenario(cost_multiplier=3.0, slippage_bps=10.0, execution_delay_days=2),
    ]

    def runner(scenario: SensitivityScenario) -> SensitivityMetrics:
        drag = 0.01 * scenario.cost_multiplier + scenario.slippage_bps / 10000
        return SensitivityMetrics(
            total_return=0.10 - drag,
            sharpe=1.0 - drag,
            max_drawdown=-0.05 - drag,
            diagnostic_pass=scenario.cost_multiplier < 3.0,
        )

    report = SensitivityAnalyzer().run(scenarios, runner)

    assert report.summary.scenario_count == 3
    assert report.summary.baseline_total_return == pytest.approx(0.09)
    assert report.summary.worst_total_return == pytest.approx(0.069)
    assert report.summary.return_degradation == pytest.approx(0.021)
    assert report.summary.pass_ratio == 2 / 3
    assert report.summary.worst_scenario == scenarios[-1]


def test_sensitivity_grid_rejects_invalid_values() -> None:
    grid = SensitivityGrid(cost_multipliers=(0.0,))

    try:
        grid.scenarios()
    except ValueError as exc:
        assert "cost multipliers" in str(exc)
    else:
        raise AssertionError("expected invalid cost multiplier to fail")
