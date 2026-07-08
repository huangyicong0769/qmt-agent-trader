from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.query_tools import query_universe_tool, set_data_lake
from qmt_agent_trader.data.storage import DataLake


def _lake_with_stock_and_etf(tmp_path) -> DataLake:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    bar = {
        "trade_date": "20260708",
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.0,
    }
    lake.write_parquet(pd.DataFrame([{**bar, "ts_code": "000001.SZ"}]), "raw", "tushare/daily")
    lake.write_parquet(
        pd.DataFrame([{**bar, "ts_code": "159259.SZ"}]),
        "raw",
        "tushare/fund_daily",
    )
    return lake


def test_query_universe_stock_excludes_etf_source(tmp_path) -> None:
    set_data_lake(_lake_with_stock_and_etf(tmp_path))

    result = query_universe_tool.run(
        {"as_of_date": "20260708", "universe_type": "stock"},
        ToolContext(run_id="universe-stock"),
    )

    assert result["status"] == "OK"
    assert result["symbols"] == ["000001.SZ"]
    assert result["metadata"]["universe_type"] == "stock"


def test_query_universe_etf_excludes_stock_source(tmp_path) -> None:
    set_data_lake(_lake_with_stock_and_etf(tmp_path))

    result = query_universe_tool.run(
        {"as_of_date": "20260708", "universe_type": "etf"},
        ToolContext(run_id="universe-etf"),
    )

    assert result["status"] == "OK"
    assert result["symbols"] == ["159259.SZ"]
    assert result["metadata"]["universe_type"] == "etf"


def test_query_universe_mixed_reports_asset_type_by_symbol(tmp_path) -> None:
    set_data_lake(_lake_with_stock_and_etf(tmp_path))

    result = query_universe_tool.run(
        {"as_of_date": "20260708", "universe_type": "mixed"},
        ToolContext(run_id="universe-mixed"),
    )

    assert set(result["symbols"]) == {"000001.SZ", "159259.SZ"}
    assert result["metadata"]["universe_type"] == "mixed"
    assert result["metadata"]["asset_type_by_symbol"] == {
        "000001.SZ": "stock",
        "159259.SZ": "etf",
    }


def test_query_universe_rejects_cyclical_theme_for_etf(tmp_path) -> None:
    set_data_lake(_lake_with_stock_and_etf(tmp_path))

    result = query_universe_tool.run(
        {
            "as_of_date": "20260708",
            "universe_type": "etf",
            "filters": {"theme": "cyclical"},
        },
        ToolContext(run_id="universe-theme-etf"),
    )

    assert result["status"] == "INVALID_REQUEST"
    assert result["domain_status"] == "INVALID_REQUEST"
    assert result["reason"] == "theme_universe_requires_stock"
