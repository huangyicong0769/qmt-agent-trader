from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.trading_calendar import (
    latest_open_session_on_or_before,
    load_open_sessions,
)


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


def test_latest_open_session_on_or_before_skips_closed_boundary(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    assert latest_open_session_on_or_before(lake, as_of="20240103") == date(
        2024, 1, 2
    )


def test_latest_open_session_requires_boundary_date_evidence(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240102", "is_open": 1},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        latest_open_session_on_or_before(lake, as_of="20240103")

    assert exc_info.value.code == "TRADING_CALENDAR_PARTIAL_COVERAGE"
    assert exc_info.value.details["missing_dates"] == ["2024-01-03"]


def test_latest_open_session_allows_closed_boundary_with_evidence(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240106", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    assert latest_open_session_on_or_before(lake, as_of="20240106") == date(
        2024, 1, 5
    )

def test_missing_trade_calendar_raises(tmp_path) -> None:
    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_open_sessions(data_lake(tmp_path), start="20240102", end="20240104")

    assert exc_info.value.code == "TRADING_CALENDAR_NOT_READY"


def test_partial_calendar_cannot_hide_missing_session(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240104", "is_open": 1},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_open_sessions(lake, start="20240102", end="20240104")

    assert exc_info.value.code == "TRADING_CALENDAR_PARTIAL_COVERAGE"
    assert exc_info.value.details["missing_dates"] == ["2024-01-03"]


def test_conflicting_calendar_states_raise(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240102", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_open_sessions(lake, start="20240102", end="20240102")

    assert exc_info.value.code == "TRADING_CALENDAR_CONFLICTING_STATE"


@pytest.mark.parametrize(
    ("cal_date", "is_open"),
    [
        ("not-a-date", 1),
        ("20240102", 2),
    ],
)
def test_invalid_calendar_values_raise(tmp_path, cal_date, is_open) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [{"exchange": "SSE", "cal_date": cal_date, "is_open": is_open}]
        ),
        "raw",
        "tushare/trade_cal",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_open_sessions(lake, start="20240102", end="20240102")

    assert exc_info.value.code == "TRADING_CALENDAR_INVALID"
