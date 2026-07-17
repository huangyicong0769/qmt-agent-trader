from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.data.trade_state import normalize_etf_opening_trade_state


def test_etf_uses_limit_source_without_stock_st_state() -> None:
    bars = pd.DataFrame(
        [
            {
                "symbol": "510300.SH",
                "trade_date": date(2024, 1, 2),
                "asset_type": "etf",
                "open": 3.5,
                "high": 3.6,
                "low": 3.4,
                "close": 3.55,
                "volume": 1000.0,
                "amount": 3500.0,
                "turnover": pd.NA,
            }
        ]
    )
    limits = pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_date": "20240102",
                "up_limit": 3.85,
                "down_limit": 3.15,
            }
        ]
    )

    observed = normalize_etf_opening_trade_state(
        bars,
        stk_limit=limits,
    )

    assert observed["st"].tolist() == [False]
    assert observed["suspended"].tolist() == [False]
    assert observed["limit_up_at_open"].tolist() == [False]
    assert observed.attrs["trade_state_quality"]["asset_type"] == "etf"
