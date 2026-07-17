from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
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
                _bar("000003.SZ", "20240103", st=True),
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

    result = _resolver(lake).build(
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


def test_resolver_rejects_missing_list_date(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([_bar("000001.SZ", "20240103")]),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "平安银行",
                    "industry": "银行",
                    "list_status": "L",
                    "list_date": float("nan"),
                }
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    spec = UniverseSpec(
        universe_id="u_nan_list_date",
        name="NaN list date",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        filters=UniverseFilters(min_listed_days=30),
        mode="snapshot",
        created_at="2026-07-09T00:00:00+08:00",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        _resolver(lake).build(spec, as_of_date="20240103")

    assert exc_info.value.code == "UNIVERSE_SECURITY_MASTER_INVALID"


def test_resolver_rejects_malformed_list_date(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([_bar("000001.SZ", "20240103")]),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "平安银行",
                    "industry": "银行",
                    "list_status": "L",
                    "list_date": "not-a-date",
                }
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    spec = UniverseSpec(
        universe_id="u_bad_list_date",
        name="Bad list date",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        filters=UniverseFilters(min_listed_days=30),
        mode="snapshot",
        created_at="2026-07-09T00:00:00+08:00",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        _resolver(lake).build(
            spec,
            as_of_date="20240103",
            include_exclusions=True,
        )

    assert exc_info.value.code == "UNIVERSE_SECURITY_MASTER_INVALID"


def test_broad_universe_uses_per_symbol_latest_asof_not_global_latest(
    tmp_path: Path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    previous_symbols = [f"{index:06d}.SZ" for index in range(1, 2001)]
    latest_symbols = previous_symbols[:3]
    lake.write_parquet(
        pd.DataFrame(
            [_bar(symbol, "20260707") for symbol in previous_symbols]
            + [_bar(symbol, "20260708") for symbol in latest_symbols]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                _stock_basic(symbol, f"股票{index}", "测试")
                for index, symbol in enumerate(previous_symbols)
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    spec = UniverseSpec(
        universe_id="u_sparse_latest",
        name="Sparse latest",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        filters=UniverseFilters(min_listed_days=0),
        mode="snapshot",
        created_at="2026-07-09T00:00:00+08:00",
    )

    result = _resolver(lake).build(spec, as_of_date="20260708", limit=2500)

    assert result["status"] == "OK"
    assert len(result["symbols"]) == 2000
    diagnostics = result["metadata"]["diagnostics"]
    assert diagnostics["latest_global_trade_date"] == "20260708"
    assert diagnostics["symbols_on_latest_global_trade_date"] == 3
    assert diagnostics["symbols_with_bar_before_as_of"] == 2000


def test_snapshot_universe_exposes_staleness_diagnostics(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    lake.write_parquet(
        pd.DataFrame(
            [
                _bar("000001.SZ", "20260708"),
                _bar("000002.SZ", "20260707"),
                _bar("000003.SZ", "20260703"),
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame([_stock_basic(symbol, symbol, "测试") for symbol in symbols]),
        "raw",
        "tushare/stock_basic",
    )
    spec = UniverseSpec(
        universe_id="u_stale",
        name="Stale bars",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        filters=UniverseFilters(min_listed_days=0),
        mode="snapshot",
        created_at="2026-07-09T00:00:00+08:00",
    )

    result = _resolver(lake).build(spec, as_of_date="20260708")

    diagnostics = result["metadata"]["diagnostics"]
    assert diagnostics["stale_symbol_count"] == 2
    assert diagnostics["max_bar_staleness_days"] == 5
    assert diagnostics["recent_bar_symbol_count"] == 3


def _bar(
    symbol: str,
    trade_date: str,
    *,
    suspended: bool = False,
    st: bool = False,
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
        "suspended": suspended,
        "st": st,
    }


def _stock_basic(symbol: str, name: str, industry: str) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "name": name,
        "industry": industry,
        "list_status": "L",
        "list_date": "20200101",
    }


def _resolver(lake: DataLake) -> UniverseResolver:
    bars = lake.read_parquet("raw", "tushare/daily")
    suspended = (
        bars["suspended"].fillna(False).astype(bool)
        if "suspended" in bars.columns
        else pd.Series(False, index=bars.index)
    )
    lake.write_parquet(
        bars.loc[suspended, ["ts_code", "trade_date"]].assign(suspend_type="S"),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        bars[["ts_code", "trade_date"]].assign(up_limit=12.0, down_limit=8.0),
        "raw",
        "tushare/stk_limit",
    )
    st_state = (
        bars["st"].fillna(False).astype(bool)
        if "st" in bars.columns
        else pd.Series(False, index=bars.index)
    )
    st_symbols = set(bars.loc[st_state, "ts_code"].astype(str))
    namechange = pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "name": "ST fixture",
                "start_date": bars.loc[bars["ts_code"] == symbol, "trade_date"].min(),
                "end_date": bars.loc[bars["ts_code"] == symbol, "trade_date"].max(),
            }
            for symbol in sorted(st_symbols)
        ],
        columns=["ts_code", "name", "start_date", "end_date"],
    )
    lake.write_parquet(namechange, "raw", "tushare/namechange")
    return UniverseResolver(lake)
