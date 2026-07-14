from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)


def test_completely_missing_open_session_aborts_before_execution() -> None:
    expected = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    )
    bars = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100.0,
                "amount": 1_000.0,
                "turnover": 0.01,
                "suspended": False,
                "st": False,
                "limit_up_at_open": False,
                "limit_down_at_open": False,
            }
            for trade_date in (expected[0], expected[2])
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        FactorRankResearchRunner(
            bars,
            FactorRankResearchConfig(
                factor_name="fixture",
                expected_trade_dates=expected,
            ),
        )

    assert exc_info.value.code == "MISSING_EXPECTED_TRADING_SESSION"
    assert exc_info.value.details["missing_dates"] == ["2024-01-03"]
