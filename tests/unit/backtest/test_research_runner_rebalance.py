from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.backtest import research_runner
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario


def _bars(days: int = 8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset in range(days):
        for index, symbol in enumerate(("A", "B", "C")):
            price = 10.0 + index
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date(2024, 1, 2) + timedelta(days=offset),
                    "open": price,
                    "close": price,
                    "suspended": False,
                    "limit_up": False,
                    "limit_down": False,
                    "st": False,
                }
            )
    return pd.DataFrame(rows)


def _run(monkeypatch, **updates):
    bars = _bars()
    factors = bars[["symbol", "trade_date"]].copy()
    factors["factor_value"] = factors["symbol"].map({"A": 3.0, "B": 2.0, "C": 1.0})
    monkeypatch.setattr(research_runner, "compute_factor_frame", lambda *_args, **_kw: factors)
    config = FactorRankResearchConfig(
        factor_name="test",
        expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
        top_n=updates.pop("top_n", 1),
        max_single_position_pct=updates.pop("max_single_position_pct", 1.0),
        initial_cash=100_000,
        **updates,
    )
    return FactorRankResearchRunner(bars, config).run(
        SensitivityScenario(
            cost_multiplier=0.0,
            slippage_bps=0.0,
            execution_delay_days=1,
            top_n=config.top_n,
            max_single_position_pct=config.max_single_position_pct,
        )
    )


def test_cash_buffer_reduces_target_investment(monkeypatch) -> None:
    result = _run(monkeypatch, top_n=2, cash_buffer_pct=0.20)
    first = result.rebalance_points[0]
    buys = [
        trade
        for trade in result.trades
        if trade.trade_date == first.trade_date and trade.side.value == "BUY"
    ]
    invested = sum(trade.reference_price * trade.quantity for trade in buys)
    assert first.equity_before * 0.75 <= invested <= first.equity_before * 0.80


def test_lower_is_better_reverses_factor_direction(monkeypatch) -> None:
    result = _run(monkeypatch, lower_is_better=True)
    first_buy = next(trade for trade in result.trades if trade.side.value == "BUY")
    assert first_buy.symbol == "C"


def test_small_planned_turnover_skips_entire_rebalance(monkeypatch) -> None:
    result = _run(monkeypatch, min_turnover_threshold=1.0)
    assert result.trades == ()
    assert all(point.skipped for point in result.rebalance_points)
    assert {point.skip_reason for point in result.rebalance_points} == {
        "below_min_turnover_threshold"
    }


def test_rank_buffer_retains_holding_inside_exit_band(monkeypatch) -> None:
    bars = _bars()
    factors = bars[["symbol", "trade_date"]].copy()
    start = factors["trade_date"].min()
    factors["factor_value"] = factors.apply(
        lambda row: (
            {"A": 3.0, "B": 2.0, "C": 1.0}[row["symbol"]]
            if row["trade_date"] <= start + timedelta(days=1)
            else {"A": 2.0, "B": 3.0, "C": 1.0}[row["symbol"]]
        ),
        axis=1,
    )
    monkeypatch.setattr(research_runner, "compute_factor_frame", lambda *_args, **_kw: factors)

    def run(rank_buffer: int):
        config = FactorRankResearchConfig(
            factor_name="test",
            expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=1.0,
            initial_cash=100_000,
            cash_buffer_pct=0.0,
            rank_buffer=rank_buffer,
        )
        return FactorRankResearchRunner(bars, config).run(
            SensitivityScenario(
                cost_multiplier=0.0,
                slippage_bps=0.0,
                execution_delay_days=1,
                top_n=1,
                max_single_position_pct=1.0,
            )
        )

    assert sum(point.exited_count for point in run(1).rebalance_points) < sum(
        point.exited_count for point in run(0).rebalance_points
    )
