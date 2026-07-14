from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from qmt_agent_trader.backtest import research_runner
from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario


def _bars(days: int = 45) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset in range(days):
        trade_date = date(2024, 1, 2) + timedelta(days=offset)
        for index, symbol in enumerate(("A", "B")):
            price = 10.0 + index
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000.0,
                    "amount": price * 1_000.0,
                    "turnover": 0.01,
                    "suspended": False,
                    "st": False,
                    "limit_up_at_open": False,
                    "limit_down_at_open": False,
                }
            )
    return pd.DataFrame(rows)


def _run(monkeypatch, *, frequency: str, symbols_by_date: dict[str, list[str]]):
    bars = _bars()
    factors = bars[["symbol", "trade_date"]].copy()
    factors["factor_value"] = factors["symbol"].map({"A": 2.0, "B": 1.0})
    monkeypatch.setattr(research_runner, "compute_factor_frame", lambda *_a, **_kw: factors)
    config = FactorRankResearchConfig(
        factor_name="fixture",
        expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
        top_n=1,
        max_single_position_pct=1.0,
        initial_cash=100_000,
        rebalance_frequency=frequency,
        symbols_by_date=symbols_by_date,
    )
    return FactorRankResearchRunner(bars, config).run(
        SensitivityScenario(
            cost_multiplier=1.0,
            slippage_bps=0.0,
            execution_delay_days=1,
            top_n=1,
            max_single_position_pct=1.0,
        )
    )


@pytest.mark.parametrize(
    ("frequency", "symbols_by_date"),
    [
        ("daily", {"20240102": ["A"], "20240107": ["A"]}),
        ("weekly", {"20240102": ["A"], "20240107": ["A"]}),
        ("monthly", {"20240102": ["A"], "20240131": ["A"]}),
    ],
)
def test_strategy_frequency_uses_as_of_universe_membership(
    monkeypatch,
    frequency,
    symbols_by_date,
) -> None:
    result = _run(
        monkeypatch,
        frequency=frequency,
        symbols_by_date=symbols_by_date,
    )

    assert any(trade.side.value == "BUY" for trade in result.trades)


def test_missing_initial_snapshot_raises(monkeypatch) -> None:
    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        _run(
            monkeypatch,
            frequency="daily",
            symbols_by_date={"20240105": ["A"]},
        )

    assert exc_info.value.code == "ROLLING_UNIVERSE_SNAPSHOT_NOT_AVAILABLE"


def test_empty_resolved_snapshot_raises(monkeypatch) -> None:
    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        _run(
            monkeypatch,
            frequency="daily",
            symbols_by_date={"20240102": []},
        )

    assert exc_info.value.code == "ROLLING_UNIVERSE_EMPTY_AS_OF_SIGNAL"
