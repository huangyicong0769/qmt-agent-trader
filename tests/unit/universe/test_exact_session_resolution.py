from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.resolver import UniverseResolver


def _write_empty_trade_state_sources(lake: DataLake) -> None:
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )


def _write_stock_basic(lake: DataLake, symbols: list[str]) -> None:
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "name": symbol,
                    "list_status": "L",
                    "list_date": "20000101",
                    "delist_date": None,
                }
                for symbol in symbols
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )


def _write_calendar_sessions(lake: DataLake, session_keys: list[str]) -> None:
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": exchange,
                    "cal_date": session_key,
                    "is_open": 1,
                }
                for session_key in session_keys
                for exchange in ("SSE", "SZSE")
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )


def _all_stock_spec() -> UniverseSpec:
    return UniverseSpec.model_validate(
        {
            "universe_id": "all_stock",
            "name": "All stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )


def test_open_market_session_without_any_bars_fails_closed(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    _write_calendar_sessions(lake, ["20240102", "20240103"])
    _write_stock_basic(lake, ["000001.SZ"])
    _write_empty_trade_state_sources(lake)
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake).build(
            _all_stock_spec(),
            mode="snapshot",
            as_of_date="20240103",
        )

    assert exc_info.value.code == "UNIVERSE_MARKET_SESSION_NOT_READY"
    assert exc_info.value.trade_date == "2024-01-03"


def test_twenty_session_metrics_use_bounded_raw_read(tmp_path, monkeypatch) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(20)]
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"{day:%Y%m%d}",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
                for day in days
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": f"{day:%Y%m%d}",
                    "is_open": 1,
                }
                for day in days
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    observed_reads: list[dict[str, object]] = []
    original = lake.read_parquet_filtered

    def recording_read(*args, **kwargs):
        if len(args) > 1 and args[1] == "tushare/daily":
            observed_reads.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(lake, "read_parquet_filtered", recording_read)

    metrics = UniverseResolver(lake)._avg_20d_metrics("20240120", ["stock"])

    assert metrics["avg_amount_20d"].tolist() == [1000.0]
    assert observed_reads == [
        {
            "start": "20240101",
            "end": "20240120",
            "columns": [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ],
        }
    ]
