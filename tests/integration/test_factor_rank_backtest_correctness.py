from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from qmt_agent_trader.backtest import research_runner
from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario


def _bars(*, symbols: int = 100, days: int = 80, missing=None) -> pd.DataFrame:
    missing = missing or set()
    rows: list[dict[str, object]] = []
    for offset in range(days):
        for index in range(symbols):
            symbol = f"{index:06d}.SZ"
            if (symbol, offset) in missing:
                continue
            price = 10.0 + index / 100 + offset / 1000
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date(2024, 1, 2) + timedelta(days=offset),
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


def _install_factor(monkeypatch, bars: pd.DataFrame) -> None:
    factors = bars[["symbol", "trade_date"]].copy()
    factors["factor_value"] = factors["symbol"].str[:6].astype(int)
    monkeypatch.setattr(research_runner, "compute_factor_frame", lambda *_a, **_k: factors)


def test_weekly_correctness_replay_emits_complete_daily_evidence(monkeypatch) -> None:
    bars = _bars()
    _install_factor(monkeypatch, bars)
    runner = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="fixture",
            expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
            top_n=10,
            max_single_position_pct=0.10,
            initial_cash=1_000_000,
            rebalance_frequency="weekly",
            cash_buffer_pct=0.10,
            rank_buffer=10,
        ),
    )
    result = runner.run(
        SensitivityScenario(
            cost_multiplier=1.0,
            slippage_bps=5.0,
            execution_delay_days=1,
            top_n=10,
            max_single_position_pct=0.10,
        )
    )
    assert len(result.equity_points) == 80
    assert result.same_trade_gross_return >= result.metrics.total_return
    assert result.data_quality.validated_valuation_dates == 80
    assert result.total_explicit_cost > 0
    assert result.total_slippage_cost > 0


def test_missing_held_bar_fails_without_completed_evidence(monkeypatch) -> None:
    bars = _bars(symbols=2, days=6, missing={("000001.SZ", 2)})
    _install_factor(monkeypatch, bars)
    runner = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="fixture",
            expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=1.0,
            initial_cash=100_000,
            cash_buffer_pct=0.0,
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
    assert exc_info.value.code == "MISSING_HELD_POSITION_BAR"
