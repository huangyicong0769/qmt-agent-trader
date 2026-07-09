from __future__ import annotations

from pathlib import Path

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseFilters, UniverseSelection, UniverseSpec
from qmt_agent_trader.universe.resolver import UniverseResolver


def test_resolver_builds_snapshot_stock_universe_with_exclusions(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                _bar("000001.SZ", "20240103"),
                _bar("000002.SZ", "20240103", suspended=True),
                _bar("000003.SZ", "20240103"),
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                _stock_basic("000001.SZ", "平安银行", "银行"),
                _stock_basic("000002.SZ", "万科A", "房地产"),
                _stock_basic("000003.SZ", "*ST测试", "软件服务"),
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    spec = UniverseSpec(
        universe_id="u_snapshot_stock",
        name="Snapshot stocks",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        filters=UniverseFilters(min_listed_days=0),
        mode="snapshot",
        created_at="2026-07-09T00:00:00+08:00",
    )

    result = UniverseResolver(lake).build(
        spec,
        as_of_date="20240103",
        include_exclusions=True,
    )

    assert result["status"] == "OK"
    assert result["mode"] == "snapshot"
    assert result["symbols"] == ["000001.SZ"]
    assert result["metadata"]["count"] == 1
    assert result["metadata"]["as_of_date"] == "20240103"
    assert result["metadata"]["fingerprint"]
    excluded = {item["symbol"]: item["reason"] for item in result["metadata"]["excluded_symbols"]}
    assert excluded["000002.SZ"] == "suspended"
    assert excluded["000003.SZ"] == "st"


def _bar(symbol: str, trade_date: str, *, suspended: bool = False) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "trade_date": trade_date,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "vol": 1000.0,
        "amount": 10000.0,
        "suspended": suspended,
    }


def _stock_basic(symbol: str, name: str, industry: str) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "name": name,
        "industry": industry,
        "list_status": "L",
        "list_date": "20200101",
    }
