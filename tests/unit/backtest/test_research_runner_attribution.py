from test_research_runner_valuation import bars_for_symbols

from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario


def _run(cost_multiplier: float, slippage_bps: float):
    bars = bars_for_symbols(["000001.SZ", "000002.SZ"], days=35)
    runner = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="momentum_20d",
            expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=1.0,
            initial_cash=100_000,
            cash_buffer_pct=0.02,
        ),
    )
    return runner.run(
        SensitivityScenario(
            cost_multiplier=cost_multiplier,
            slippage_bps=slippage_bps,
            execution_delay_days=1,
            top_n=1,
            max_single_position_pct=1.0,
        )
    )


def test_default_cost_run_exposes_same_trade_cost_drag() -> None:
    zero = _run(0.0, 0.0)
    default = _run(1.0, 5.0)
    assert default.metrics.total_return < zero.metrics.total_return
    assert default.total_explicit_cost > 0
    assert default.total_slippage_cost > 0
    assert default.same_trade_gross_return > default.metrics.total_return


def test_rebalance_points_expose_selection_overlap() -> None:
    result = _run(0.0, 0.0)
    assert any(point.selection_jaccard is not None for point in result.rebalance_points)
    assert 0.0 <= result.average_top_n_overlap <= 1.0
