from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.trade_state import normalize_stock_opening_trade_state


def bars(open_price: float, close_price: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "open": open_price,
                "close": close_price,
            }
        ]
    )


def limits() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        ]
    )


def test_close_limit_does_not_block_opening_buy() -> None:
    result = normalize_stock_opening_trade_state(
        bars(10.0, 11.0),
        suspend=pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        stk_limit=limits(),
        namechange=pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
    )
    assert not bool(result.loc[0, "limit_up_at_open"])


def test_open_at_upper_limit_blocks_opening_buy() -> None:
    result = normalize_stock_opening_trade_state(
        bars(11.0, 10.5),
        suspend=pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        stk_limit=limits(),
        namechange=pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
    )
    assert bool(result.loc[0, "limit_up_at_open"])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("up_limit", None),
        ("up_limit", 0.0),
        ("up_limit", float("inf")),
        ("down_limit", None),
        ("down_limit", -1.0),
    ],
)
def test_invalid_limit_price_fails_closed(column: str, value: float | None) -> None:
    source = limits()
    source.loc[0, column] = value
    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        normalize_stock_opening_trade_state(
            bars(10.0, 10.0),
            suspend=pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
            stk_limit=source,
            namechange=pd.DataFrame(
                columns=["ts_code", "name", "start_date", "end_date"]
            ),
        )
    assert exc_info.value.code == "INVALID_TRADE_STATE_SOURCE"
