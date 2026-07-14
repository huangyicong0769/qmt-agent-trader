import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake


def _lake_with_daily(tmp_path) -> DataLake:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1_000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )
    return lake


def test_missing_trade_state_sources_fail_closed(tmp_path) -> None:
    lake = _lake_with_daily(tmp_path)

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_daily_bars(
            lake,
            start="20240102",
            end="20240102",
            symbols=["000001.SZ"],
        )

    assert exc_info.value.code == "TRADE_STATE_SOURCE_NOT_READY"
    assert set(exc_info.value.details["missing_datasets"]) == {
        "tushare/suspend_d",
        "tushare/stk_limit",
        "tushare/namechange",
    }


def test_missing_stk_limit_row_fails_closed(tmp_path) -> None:
    lake = _lake_with_daily(tmp_path)
    _write_sparse_state_sources(lake)
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240102",
                    "up_limit": 20.0,
                    "down_limit": 18.0,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_daily_bars(
            lake,
            start="20240102",
            end="20240102",
            symbols=["000001.SZ"],
        )

    assert exc_info.value.code == "TRADE_STATE_PARTIAL_COVERAGE"
    assert exc_info.value.details["field"] == "limit_up_down"


def test_complete_trade_state_sources_produce_boolean_evidence(tmp_path) -> None:
    lake = _lake_with_daily(tmp_path)
    _write_sparse_state_sources(lake)
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

    bars = load_daily_bars(
        lake,
        start="20240102",
        end="20240102",
        symbols=["000001.SZ"],
    )

    assert bars[["suspended", "limit_up", "limit_down", "st"]].isna().sum().sum() == 0
    assert bars.attrs["trade_state_quality"]["limit_up"]["complete"] is True


def _write_sparse_state_sources(lake: DataLake) -> None:
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )
