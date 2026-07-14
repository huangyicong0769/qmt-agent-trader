from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def lake(tmp_path: Path) -> DataLake:
    return DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")


@pytest.fixture
def registry(lake: DataLake, tmp_path: Path):
    return build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )


def test_run_backtest_rejects_removed_legacy_universe_name(registry, lake: DataLake) -> None:
    _write_backtest_bars(lake)

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240210",
            "universe": "cyclical",
        },
        ToolContext(run_id="legacy-universe-name"),
    )

    assert result["status"] == "INVALID_REQUEST"
    assert result["reason"] == "LEGACY_UNIVERSE_NAME_REMOVED"


def test_snapshot_backtest_uses_one_fixed_resolved_symbol_set(
    registry,
    lake: DataLake,
) -> None:
    _write_backtest_bars(lake)
    _write_stock_basic(lake, listed_c_date="20240201")

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240210",
            "as_of_date": "20240110",
            "universe_mode": "snapshot",
            "universe_spec": _stock_universe_spec(mode="snapshot"),
            "top_n": 1,
        },
        ToolContext(run_id="snapshot-backtest"),
    )

    assert result["status"] == "completed"
    assert result["universe_mode"] == "snapshot"
    assert result["symbols_source"] == "universe_snapshot"
    assert result["universe_resolve_dates"] == ["20240110"]
    assert result["symbols_count"] == 2
    assert result["symbols_sample"] == ["000001.SZ", "000002.SZ"]
    assert result["universe_spec_fingerprint"]

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (Path(result["report_path"]).parent / ".manifests").glob("*.json")
    ]
    assert any(
        manifest["relative_path"] == Path(result["report_path"]).name
        and manifest["related_run_id"] == result["run_id"]
        for manifest in manifests
    )
    assert report["config"]["universe_mode"] == "snapshot"
    assert report["config"]["symbols"] == ["000001.SZ", "000002.SZ"]
    assert report["config"]["symbols_by_date"] is None


def _stock_universe_spec(*, mode: str) -> dict[str, object]:
    return {
        "universe_id": f"u_{mode}_stocks",
        "name": f"{mode.title()} stocks",
        "source": "agent_generated",
        "asset_types": ["stock"],
        "selection": {"mode": "all"},
        "filters": {"min_listed_days": 0},
        "mode": mode,
        "rebalance_frequency": "daily",
        "created_at": "2026-07-09T00:00:00+08:00",
    }


def _write_backtest_bars(lake: DataLake) -> None:
    rows: list[dict[str, object]] = []
    start = date(2024, 1, 1)
    for offset in range(41):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        rows.extend(
            [
                _bar("000001.SZ", trade_date, 10.0 + offset * 0.05),
                _bar("000002.SZ", trade_date, 12.0 + offset * 0.03),
                _bar("000003.SZ", trade_date, 8.0 + offset * 0.30),
            ]
        )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": item, "is_open": 1}
                for item in sorted({str(row["trade_date"]) for row in rows})
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )


def _write_stock_basic(lake: DataLake, *, listed_c_date: str) -> None:
    lake.write_parquet(
        pd.DataFrame(
            [
                _stock_basic("000001.SZ", "股票A", "20200101"),
                _stock_basic("000002.SZ", "股票B", "20200101"),
                _stock_basic("000003.SZ", "股票C", listed_c_date),
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )


def _bar(symbol: str, trade_date: str, close: float) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "trade_date": trade_date,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "vol": 1000.0,
        "amount": close * 1000,
    }


def _stock_basic(symbol: str, name: str, list_date: str) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "name": name,
        "industry": "软件服务",
        "list_status": "L",
        "list_date": list_date,
    }
