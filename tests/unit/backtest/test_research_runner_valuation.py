from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario


def bars_for_symbols(
    symbols: list[str],
    *,
    days: int,
    missing: set[tuple[str, int]] | None = None,
    opens: dict[tuple[str, int], float] | None = None,
    closes: dict[tuple[str, int], float] | None = None,
) -> pd.DataFrame:
    missing = missing or set()
    opens = opens or {}
    closes = closes or {}
    start = date(2024, 1, 2)
    rows: list[dict[str, object]] = []
    for offset in range(days):
        trade_date = start + timedelta(days=offset)
        for symbol_index, symbol in enumerate(symbols):
            if (symbol, offset) in missing:
                continue
            default_price = 10.0 + symbol_index
            open_price = opens.get((symbol, offset), default_price)
            close_price = closes.get((symbol, offset), open_price)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": open_price,
                    "high": max(open_price, close_price),
                    "low": min(open_price, close_price),
                    "close": close_price,
                    "volume": 1_000_000,
                    "amount": 10_000_000,
                    "turnover": 0.01,
                    "suspended": False,
                    "limit_up_at_open": False,
                    "limit_down_at_open": False,
                    "st": False,
                }
            )
    return pd.DataFrame(rows)


def run_zero_cost_top_one(
    bars: pd.DataFrame,
    *,
    rebalance_frequency: str = "daily",
):
    runner = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="momentum_20d",
            expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=1.0,
            initial_cash=100_000,
            rebalance_frequency=rebalance_frequency,
        ),
    )
    return runner.run(
        SensitivityScenario(
            cost_multiplier=1.0,
            slippage_bps=0.0,
            execution_delay_days=1,
            top_n=1,
            max_single_position_pct=1.0,
        )
    )


def test_missing_held_bar_aborts_backtest() -> None:
    bars = bars_for_symbols(
        ["000001.SZ", "000002.SZ"],
        days=25,
        missing={("000001.SZ", 22)},
    )
    runner = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="momentum_20d",
            expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=1.0,
            initial_cash=100_000,
        ),
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        runner.run(
            SensitivityScenario(
                cost_multiplier=1.0,
                slippage_bps=0.0,
                execution_delay_days=1,
                top_n=1,
                max_single_position_pct=1.0,
            )
        )

    error = exc_info.value
    assert error.code == "MISSING_HELD_POSITION_BAR"
    assert error.trade_date == "2024-01-24"
    assert error.symbols == ("000001.SZ",)


def test_target_quantity_does_not_use_execution_day_close() -> None:
    common_opens = {("000001.SZ", 21): 10.0, ("000002.SZ", 21): 20.0}
    low_close = bars_for_symbols(
        ["000001.SZ", "000002.SZ"],
        days=24,
        opens=common_opens,
        closes={("000001.SZ", 21): 1.0, ("000002.SZ", 21): 2.0},
    )
    high_close = bars_for_symbols(
        ["000001.SZ", "000002.SZ"],
        days=24,
        opens=common_opens,
        closes={("000001.SZ", 21): 100.0, ("000002.SZ", 21): 200.0},
    )

    low_result = run_zero_cost_top_one(low_close)
    high_result = run_zero_cost_top_one(high_close)

    low_first_buy = next(trade for trade in low_result.trades if trade.side.value == "BUY")
    high_first_buy = next(trade for trade in high_result.trades if trade.side.value == "BUY")
    assert low_first_buy.quantity == high_first_buy.quantity
