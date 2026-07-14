import pandas as pd
from test_research_runner_valuation import (
    bars_for_symbols,
    run_zero_cost_top_one,
)

from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario


def test_equity_points_cover_every_trade_date_after_initialization() -> None:
    bars = bars_for_symbols(["000001.SZ", "000002.SZ"], days=30)
    result = run_zero_cost_top_one(bars, rebalance_frequency="weekly")

    all_dates = sorted(pd.to_datetime(bars["trade_date"]).dt.date.unique())
    result_dates = [point.trade_date for point in result.equity_points]

    assert result_dates == [f"{item:%Y-%m-%d}" for item in all_dates]
    assert len(result.rebalance_points) < len(result.equity_points)


def test_pre_start_bars_are_factor_inputs_not_equity_dates() -> None:
    bars = bars_for_symbols(["000001.SZ"], days=30)
    all_dates = tuple(sorted(bars["trade_date"].unique()))
    expected = all_dates[-2:]

    result = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="momentum_20d",
            expected_trade_dates=expected,
            top_n=1,
            max_single_position_pct=1.0,
        ),
    ).run(
        SensitivityScenario(
            execution_delay_days=1,
            top_n=1,
            max_single_position_pct=1.0,
        )
    )

    assert [point.trade_date for point in result.equity_points] == [
        f"{day:%Y-%m-%d}" for day in expected
    ]
