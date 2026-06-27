from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.query_tools import (
    list_data_catalog_tool,
    query_bars_tool,
    set_data_lake,
)
from qmt_agent_trader.data.storage import DataLake


def test_list_data_catalog_hides_legacy_tushare_batches(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    frame = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102"}])
    lake.write_parquet(frame, "raw", "tushare_daily")
    lake.write_parquet(frame, "raw", "tushare_daily_adjusted")
    lake.write_parquet(frame, "raw", "tushare_daily_20240101_20240103")
    lake.write_parquet(frame, "raw", "tushare_suspend_20240101_20240103")
    lake.write_parquet(frame, "raw", "tushare_stk_limit_20240101_20240103")
    lake.write_parquet(frame, "gold", "factor_momentum_20d_20240102")
    set_data_lake(lake)

    result = list_data_catalog_tool.run({}, ToolContext(run_id="catalog"))

    assert result["status"] == "ok"
    assert "tushare_daily" in result["layers"]["raw"]
    assert "tushare_daily_adjusted" in result["layers"]["raw"]
    assert "factor_momentum_20d_20240102" in result["layers"]["gold"]
    assert "tushare_daily_20240101_20240103" not in result["layers"]["raw"]
    assert "tushare_suspend_20240101_20240103" not in result["layers"]["raw"]
    assert "tushare_stk_limit_20240101_20240103" not in result["layers"]["raw"]


def test_query_bars_filters_symbol_alias_and_includes_symbol(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "159259.SZ",
                    "trade_date": "20250828",
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                    "vol": 100,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20250828",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vol": 200,
                },
            ]
        ),
        "raw",
        "tushare_fund_daily",
    )
    set_data_lake(lake)

    result = query_bars_tool.run(
        {"symbol": "159259", "start_date": "20250801", "end_date": "20250831"},
        ToolContext(run_id="bars"),
    )

    assert result["metadata"]["requested_symbols"] == ["159259.SZ"]
    assert result["metadata"]["returned"] == 1
    assert result["metadata"]["requested_start_date"] == "20250801"
    assert result["metadata"]["requested_end_date"] == "20250831"
    assert result["metadata"]["actual_start_date"] == "2025-08-28"
    assert result["metadata"]["actual_end_date"] == "2025-08-28"
    assert result["metadata"]["data_freshness"] == "stale_vs_requested_end"
    assert result["rows"] == [
        {
            "symbol": "159259.SZ",
            "trade_date": pd.Timestamp("2025-08-28").date(),
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "volume": 100,
        }
    ]


def test_query_bars_keeps_identity_fields_when_fields_are_requested(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "159259.SZ",
                    "trade_date": "20260626",
                    "open": 1.7,
                    "high": 1.8,
                    "low": 1.69,
                    "close": 1.739,
                    "vol": 100,
                    "amount": 173900,
                }
            ]
        ),
        "raw",
        "tushare_fund_daily",
    )
    set_data_lake(lake)

    result = query_bars_tool.run(
        {
            "symbol": "159259",
            "start_date": "20260601",
            "end_date": "20260626",
            "fields": ["close", "volume"],
        },
        ToolContext(run_id="bars-fields"),
    )

    assert result["metadata"]["identity_fields_forced"] is True
    assert result["rows"] == [
        {
            "symbol": "159259.SZ",
            "trade_date": pd.Timestamp("2026-06-26").date(),
            "close": 1.739,
            "volume": 100,
        }
    ]


def test_query_bars_filters_code_alias_without_returning_market_head(tmp_path) -> None:
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
                    "vol": 200,
                }
            ]
        ),
        "raw",
        "tushare_daily",
    )
    set_data_lake(lake)

    result = query_bars_tool.run(
        {"code": "159259.SZ", "start_date": "20240101", "end_date": "20240131"},
        ToolContext(run_id="bars-empty"),
    )

    assert result["rows"] == []
    assert result["metadata"]["returned"] == 0
    assert result["metadata"]["reason"] == "no matching bars"
