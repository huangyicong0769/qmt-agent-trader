import pandas as pd

from qmt_agent_trader.data.bars import (
    enrich_trade_states,
    load_daily_bars,
    normalize_tushare_daily,
)
from qmt_agent_trader.data.storage import DataLake


def test_normalize_tushare_daily_ignores_empty_marker_column() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "vol": 1000.0,
                "amount": 10000.0,
                "_empty": None,
            }
        ]
    )

    bars = normalize_tushare_daily(frame)

    assert bars.iloc[0]["symbol"] == "000001.SZ"
    assert str(bars.iloc[0]["trade_date"]) == "2024-01-02"
    assert "turnover" in bars.columns


def test_load_daily_bars_ignores_legacy_daily_batches(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                }
            ]
        ),
        "raw",
        "tushare_daily_20240101_20240103",
    )

    bars = load_daily_bars(lake)

    assert bars.empty


def test_load_daily_bars_reads_stable_daily_dataset_only(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = {
        "ts_code": "000001.SZ",
        "trade_date": "20240102",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.0,
    }
    stable = dict(legacy)
    stable["close"] = 10.5

    lake.write_parquet(pd.DataFrame([legacy]), "raw", "tushare_daily_20240101_20240103")
    lake.write_parquet(pd.DataFrame([stable]), "raw", "tushare_daily")

    bars = load_daily_bars(lake)

    assert len(bars) == 1
    assert bars.iloc[0]["close"] == 10.5


def test_enrich_trade_states_uses_suspend_limit_and_st_sources() -> None:
    bars = normalize_tushare_daily(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 9.5,
                    "close": 10.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240102",
                    "open": 4.5,
                    "high": 5.0,
                    "low": 4.5,
                    "close": 4.5,
                },
                {
                    "ts_code": "000003.SZ",
                    "trade_date": "20240102",
                    "open": 8.0,
                    "high": 8.1,
                    "low": 7.9,
                    "close": 8.0,
                },
                {
                    "ts_code": "000004.SZ",
                    "trade_date": "20240102",
                    "open": 12.0,
                    "high": 12.1,
                    "low": 11.8,
                    "close": 12.0,
                },
            ]
        )
    )

    enriched = enrich_trade_states(
        bars,
        suspend=pd.DataFrame(
            [{"ts_code": "000003.SZ", "trade_date": "20240102", "suspend_type": "S"}]
        ),
        stk_limit=pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "up_limit": 10.0,
                    "down_limit": 9.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240102",
                    "up_limit": 5.5,
                    "down_limit": 4.5,
                },
            ]
        ),
        stock_basic=pd.DataFrame([{"ts_code": "000004.SZ", "name": "*ST Example"}]),
        namechange=pd.DataFrame(
            [
                {
                    "ts_code": "000003.SZ",
                    "name": "ST Historical",
                    "start_date": "20230101",
                    "end_date": "",
                }
            ]
        ),
    ).set_index("symbol")

    assert enriched.loc["000001.SZ", "limit_up"]
    assert enriched.loc["000002.SZ", "limit_down"]
    assert enriched.loc["000003.SZ", "suspended"]
    assert enriched.loc["000003.SZ", "st"]
    assert enriched.loc["000004.SZ", "st"]


def test_load_daily_bars_enriches_trade_states_from_lake(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 9.5,
                    "close": 10.0,
                }
            ]
        ),
        "raw",
        "tushare_daily",
    )
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102", "suspend_type": "S"}]),
        "raw",
        "tushare_suspend",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "up_limit": 10.0,
                    "down_limit": 9.0,
                }
            ]
        ),
        "raw",
        "tushare_stk_limit",
    )

    bars = load_daily_bars(lake)

    assert bars.iloc[0]["suspended"]
    assert bars.iloc[0]["limit_up"]
