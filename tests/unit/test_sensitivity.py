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


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        (SensitivityScenario(cost_multiplier=0), "cost_multiplier must be positive"),
        (SensitivityScenario(slippage_bps=-1), "slippage_bps must be non-negative"),
        (
            SensitivityScenario(execution_delay_days=0),
            "execution_delay_days must be at least one",
        ),
        (SensitivityScenario(top_n=0), "top_n must be positive when provided"),
        (
            SensitivityScenario(max_single_position_pct=2),
            "max_single_position_pct must be in (0, 1]",
        ),
    ],
)
def test_factor_rank_scenario_rejects_invalid_values(scenario, message) -> None:
    with pytest.raises(ValueError) as exc_info:
        scenario.validate_for_factor_rank()

    assert str(exc_info.value) == message


def test_sensitivity_grid_rejects_zero_execution_delay() -> None:
    with pytest.raises(ValueError, match="execution delay days must be positive"):
        SensitivityGrid(execution_delay_days=(0,)).scenarios()
