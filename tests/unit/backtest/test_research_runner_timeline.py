import pandas as pd

from test_research_runner_valuation import (
    bars_for_symbols,
    run_zero_cost_top_one,
)


def test_equity_points_cover_every_trade_date_after_initialization() -> None:
    bars = bars_for_symbols(["000001.SZ", "000002.SZ"], days=30)
    result = run_zero_cost_top_one(bars, rebalance_frequency="weekly")

    all_dates = sorted(pd.to_datetime(bars["trade_date"]).dt.date.unique())
    result_dates = [point.trade_date for point in result.equity_points]

    assert result_dates == [f"{item:%Y-%m-%d}" for item in all_dates]
    assert len(result.rebalance_points) < len(result.equity_points)
