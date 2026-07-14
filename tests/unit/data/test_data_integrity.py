from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.bars import normalize_tushare_daily
from qmt_agent_trader.data.integrity import require_unique_symbol_dates


@pytest.mark.parametrize("second_close", [10.0, 11.0])
def test_symbol_date_duplicates_are_always_rejected(second_close: float) -> None:
    frame = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": date(2024, 1, 2), "close": 10.0},
            {
                "ts_code": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "close": second_close,
            },
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        require_unique_symbol_dates(
            frame,
            symbol_column="ts_code",
            date_column="trade_date",
            code="DUPLICATE_SYMBOL_DATE_BAR",
            field="raw/tushare/daily",
        )

    assert exc_info.value.code == "DUPLICATE_SYMBOL_DATE_BAR"
    assert exc_info.value.symbols == ("000001.SZ",)
    assert exc_info.value.details["duplicate_key_count"] == 1


def test_unique_symbol_dates_pass() -> None:
    frame = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": date(2024, 1, 2)},
            {"symbol": "A", "trade_date": date(2024, 1, 3)},
        ]
    )
    require_unique_symbol_dates(
        frame,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_SYMBOL_DATE_BAR",
        field="bars",
    )


def test_daily_normalization_does_not_hide_duplicates() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.1,
                "high": 10.6,
                "low": 9.6,
                "close": 10.2,
            },
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        normalize_tushare_daily(raw)

    assert exc_info.value.code == "DUPLICATE_SYMBOL_DATE_BAR"
