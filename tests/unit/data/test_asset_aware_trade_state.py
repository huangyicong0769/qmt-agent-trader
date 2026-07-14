import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake


def test_etf_does_not_reuse_stock_limit_source(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20240102",
                    "open": 3.5,
                    "high": 3.6,
                    "low": 3.4,
                    "close": 3.55,
                    "vol": 100.0,
                    "amount": 350.0,
                }
            ]
        ),
        "raw",
        "tushare/fund_daily",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_daily_bars(
            lake,
            start="20240102",
            end="20240102",
            symbols=["510300.SH"],
        )

    assert exc_info.value.code == "UNSUPPORTED_ETF_TRADE_STATE_MODEL"
    assert exc_info.value.symbols == ("510300.SH",)


def test_stock_rows_have_asset_type_and_complete_state(tmp_path) -> None:
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

    bars = load_daily_bars(lake, start="20240102", end="20240102")

    assert bars["asset_type"].tolist() == ["stock"]
    assert (
        not bars[["suspended", "st", "limit_up_at_open", "limit_down_at_open"]].isna().any().any()
    )
