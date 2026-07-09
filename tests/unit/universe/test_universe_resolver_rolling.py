from __future__ import annotations

from pathlib import Path

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseFilters, UniverseSelection, UniverseSpec
from qmt_agent_trader.universe.resolver import UniverseResolver


def test_resolver_builds_rolling_universe_per_rebalance_date(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                _bar("000001.SZ", "20240102"),
                _bar("000002.SZ", "20240102"),
                _bar("000001.SZ", "20240103", st=True),
                _bar("000002.SZ", "20240103"),
                _bar("000003.SZ", "20240103"),
                _bar("000001.SZ", "20240104", st=True),
                _bar("000002.SZ", "20240104", suspended=True),
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                _stock_basic("000001.SZ", "股票A"),
                _stock_basic("000002.SZ", "股票B"),
                _stock_basic("000003.SZ", "股票C"),
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    spec = UniverseSpec(
        universe_id="u_rolling_stock",
        name="Rolling stocks",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        filters=UniverseFilters(min_listed_days=0),
        mode="rolling",
        rebalance_frequency="daily",
        created_at="2026-07-09T00:00:00+08:00",
    )

    result = UniverseResolver(lake).build(
        spec,
        mode="rolling",
        start_date="20240102",
        end_date="20240104",
        include_exclusions=True,
    )

    assert result["status"] == "OK"
    assert result["mode"] == "rolling"
    assert result["rolling_symbols"] == {
        "20240102": ["000001.SZ", "000002.SZ"],
        "20240103": ["000002.SZ", "000003.SZ"],
        "20240104": ["000003.SZ"],
    }
    assert result["metadata"]["empty_dates"] == []
    assert result["metadata"]["min_count"] == 1
    assert result["metadata"]["max_count"] == 2
    assert result["metadata"]["mean_count"] == 5 / 3
    assert result["metadata"]["changed_dates"] == 2
    diagnostics = result["metadata"]["diagnostics_by_date"]["20240104"]
    assert diagnostics["stale_symbol_count"] == 1
    assert diagnostics["symbols_on_latest_global_trade_date"] == 2
    exclusions = result["metadata"]["excluded_symbols_by_date"]
    assert {item["symbol"]: item["reason"] for item in exclusions["20240103"]}["000001.SZ"] == "st"
    assert (
        {item["symbol"]: item["reason"] for item in exclusions["20240104"]}["000002.SZ"]
        == "suspended"
    )


def _bar(
    symbol: str,
    trade_date: str,
    *,
    st: bool = False,
    suspended: bool = False,
) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "trade_date": trade_date,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "vol": 1000.0,
        "amount": 10000.0,
        "st": st,
        "suspended": suspended,
    }


def _stock_basic(symbol: str, name: str) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "name": name,
        "industry": "软件服务",
        "list_status": "L",
        "list_date": "20200101",
    }
