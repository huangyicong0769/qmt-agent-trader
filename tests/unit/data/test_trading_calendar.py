from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.trading_calendar import load_open_sessions


def data_lake(tmp_path) -> DataLake:
    return DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "test.duckdb")


def test_load_open_sessions_uses_trade_cal_not_observed_bars(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240104", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    assert load_open_sessions(lake, start="20240102", end="20240104") == (
        date(2024, 1, 2),
        date(2024, 1, 3),
    )


def test_missing_trade_calendar_raises(tmp_path) -> None:
    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_open_sessions(data_lake(tmp_path), start="20240102", end="20240104")

    assert exc_info.value.code == "TRADING_CALENDAR_NOT_READY"
