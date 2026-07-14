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


def _bars(days: int, symbols: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": date(2024, 1, 2) + timedelta(days=offset),
                "open": 10.0 + index,
                "close": 10.0 + index,
            }
            for offset in range(days)
            for index, symbol in enumerate(symbols)
        ]
    )


def _scenario() -> SensitivityScenario:
    return SensitivityScenario(
        cost_multiplier=0.0,
        slippage_bps=0.0,
        execution_delay_days=1,
        top_n=1,
        max_single_position_pct=1.0,
    )


def _runner(
    monkeypatch,
    bars: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    symbols_by_date: dict[str, list[str]] | None = None,
) -> FactorRankResearchRunner:
    monkeypatch.setattr(
        research_runner,
        "compute_factor_frame",
        lambda *_args, **_kwargs: factors,
    )
    return FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="fixture",
            expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=1.0,
            initial_cash=100_000,
            symbols_by_date=symbols_by_date,
        ),
    )


def test_all_null_factor_signals_fail_closed(monkeypatch) -> None:
    bars = _bars(3)
    factors = bars[["symbol", "trade_date"]].copy()
    factors["factor_value"] = float("nan")

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _runner(monkeypatch, bars, factors).run(_scenario())

    assert exc_info.value.code == "NO_EXECUTABLE_FACTOR_SIGNALS"
    assert set(exc_info.value.details["unavailable_signals"].values()) == {
        "factor_signal_all_null"
    }


def test_warmup_signal_gaps_are_recorded_and_valuation_is_complete(monkeypatch) -> None:
    bars = _bars(4)
    factors = bars[["symbol", "trade_date"]].copy()
    first_date = factors["trade_date"].min()
    factors["factor_value"] = factors.apply(
        lambda row: (
            float("nan")
            if row["trade_date"] == first_date
            else {"A": 2.0, "B": 1.0}[row["symbol"]]
        ),
        axis=1,
    )

    result = _runner(monkeypatch, bars, factors).run(_scenario())

    assert result.scheduled_rebalance_count == 3
    assert result.available_signal_count == 2
    assert result.signal_unavailable_count == 1
    assert result.rebalance_points[0].skipped is True
    assert result.rebalance_points[0].skip_reason == "factor_signal_all_null"
    assert len(result.equity_points) == 4
    assert [point.trade_date for point in result.equity_points] == [
        f"{item:%Y-%m-%d}" for item in sorted(bars["trade_date"].unique())
    ]


def test_signal_without_delayed_execution_session_fails_closed(monkeypatch) -> None:
    bars = _bars(1)
    factors = bars[["symbol", "trade_date"]].copy()
    factors["factor_value"] = 1.0

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _runner(monkeypatch, bars, factors).run(_scenario())

    assert exc_info.value.code == "NO_EXECUTION_SESSION_AFTER_SIGNAL"


def test_signal_empty_after_universe_filter_is_recorded(monkeypatch) -> None:
    bars = _bars(4)
    factors = bars[bars["symbol"] == "B"][["symbol", "trade_date"]].copy()
    first_date = factors["trade_date"].min()
    factors = factors[factors["trade_date"] != first_date]
    factors["factor_value"] = 1.0
    factors = pd.concat(
        [
            factors,
            pd.DataFrame(
                [
                    {
                        "symbol": "A",
                        "trade_date": sorted(factors["trade_date"].unique())[-2],
                        "factor_value": 2.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = _runner(
        monkeypatch,
        bars,
        factors,
        symbols_by_date={"20240102": ["A"]},
    ).run(_scenario())

    assert [point.skip_reason for point in result.rebalance_points if point.skipped] == [
        "factor_signal_date_missing",
        "factor_signal_empty_after_universe_filter",
    ]
    assert len(result.equity_points) == 4
