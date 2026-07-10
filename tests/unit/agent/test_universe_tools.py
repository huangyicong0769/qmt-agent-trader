from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def lake(tmp_path: Path) -> DataLake:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                _bar("000001.SZ", "20240102"),
                _bar("000002.SZ", "20240102"),
                _bar("000001.SZ", "20240103"),
                _bar("000002.SZ", "20240103", suspended=True),
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
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    return lake


@pytest.fixture
def registry(lake: DataLake, tmp_path: Path):
    return build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )


def test_create_universe_spec_supports_snapshot_and_rolling(registry) -> None:
    snapshot = registry.run_tool(
        "create_universe_spec",
        {
            "name": "All stocks snapshot",
            "source": "agent_generated",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "mode": "snapshot",
        },
        ToolContext(run_id="create-snapshot"),
    )
    rolling = registry.run_tool(
        "create_universe_spec",
        {
            "name": "All stocks rolling",
            "source": "agent_generated",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "mode": "rolling",
            "rebalance_frequency": "daily",
        },
        ToolContext(run_id="create-rolling"),
    )

    assert snapshot["status"] == "created"
    assert snapshot["universe_spec"]["mode"] == "snapshot"
    assert snapshot["research_only"] is True
    assert snapshot["live_trading_allowed"] is False
    assert rolling["status"] == "created"
    assert rolling["universe_spec"]["mode"] == "rolling"


def test_validate_universe_spec_rejects_arbitrary_rules(registry) -> None:
    result = registry.run_tool(
        "validate_universe_spec",
        {
            "universe_spec": {
                "universe_id": "u_bad",
                "name": "Bad rule",
                "source": "agent_generated",
                "asset_types": ["stock"],
                "selection": {
                    "mode": "composite",
                    "rules": [
                        {
                            "field": "amount",
                            "operator": "python_eval",
                            "value": "__import__('os').system('whoami')",
                        }
                    ],
                },
                "mode": "rolling",
                "rebalance_frequency": "daily",
                "created_at": "2026-07-09T00:00:00+08:00",
            }
        },
        ToolContext(run_id="validate-bad-rule"),
    )

    assert result["status"] == "INVALID_REQUEST"
    assert result["normalized_spec"] is None
    assert "operator" in result["errors"][0]["message"]


def test_build_universe_resolves_snapshot_and_rolling(registry) -> None:
    spec = _stock_spec()

    snapshot = registry.run_tool(
        "build_universe",
        {"universe_spec": spec, "as_of_date": "20240102", "mode": "snapshot"},
        ToolContext(run_id="build-snapshot"),
    )
    rolling = registry.run_tool(
        "build_universe",
        {
            "universe_spec": {**spec, "mode": "rolling"},
            "mode": "rolling",
            "start_date": "20240102",
            "end_date": "20240103",
            "rebalance_frequency": "daily",
        },
        ToolContext(run_id="build-rolling"),
    )

    assert snapshot["status"] == "OK"
    assert snapshot["mode"] == "snapshot"
    assert snapshot["symbols"] == ["000001.SZ", "000002.SZ"]
    assert rolling["status"] == "OK"
    assert rolling["mode"] == "rolling"
    assert rolling["rolling_symbols"] == {
        "20240102": ["000001.SZ", "000002.SZ"],
        "20240103": ["000001.SZ"],
    }
    assert rolling["metadata"]["empty_dates"] == []


def test_save_list_and_inspect_universe_spec_with_preview(registry) -> None:
    spec = _stock_spec("u_saved")

    saved = registry.run_tool(
        "save_universe_spec",
        {"universe_spec": spec},
        ToolContext(run_id="save-universe"),
    )
    listed = registry.run_tool(
        "list_universes",
        {"source": "agent_generated", "asset_type": "stock", "mode": "snapshot"},
        ToolContext(run_id="list-universes"),
    )
    inspected = registry.run_tool(
        "inspect_universe",
        {"universe_id": "u_saved", "preview": True, "as_of_date": "20240102"},
        ToolContext(run_id="inspect-universe"),
    )

    assert saved["status"] == "saved"
    assert saved["universe_spec"]["research_only"] is True
    assert saved["universe_spec"]["live_trading_allowed"] is False
    assert [item["universe_id"] for item in listed["universes"]] == ["u_saved"]
    assert inspected["status"] == "OK"
    assert inspected["preview"]["symbols"] == ["000001.SZ", "000002.SZ"]


def test_query_universe_rejects_removed_legacy_theme_filter(registry) -> None:
    result = registry.run_tool(
        "query_universe",
        {"as_of_date": "20240102", "filters": {"theme": "cyclical"}},
        ToolContext(run_id="legacy-theme-filter"),
    )

    assert result["status"] == "INVALID_REQUEST"
    assert result["reason"] == "LEGACY_THEME_FILTER_REMOVED"
    assert result["suggested_next_tools"] == [
        "create_universe_spec",
        "validate_universe_spec",
        "build_universe",
    ]


def _stock_spec(universe_id: str = "u_tool_stock") -> dict[str, object]:
    return {
        "universe_id": universe_id,
        "name": "Tool stock universe",
        "source": "agent_generated",
        "asset_types": ["stock"],
        "selection": {"mode": "all"},
        "filters": {"min_listed_days": 0},
        "mode": "snapshot",
        "created_at": "2026-07-09T00:00:00+08:00",
    }


def test_list_universes_reports_degraded_corrupt_records(registry, lake: DataLake) -> None:
    root = lake.root.parent / "registries" / "universes"
    root.mkdir(parents=True, exist_ok=True)
    (root / "broken.json").write_text("{broken", encoding="utf-8")
    result = registry.run_tool("list_universes", {}, ToolContext(run_id="run_diag"))
    assert result["status"] == "DEGRADED"
    assert len(result["diagnostics"]) == 1


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
