from __future__ import annotations

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
    return DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "test.duckdb")


@pytest.fixture
def registry(lake: DataLake, tmp_path: Path):
    return build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )


def test_backtest_requires_explicit_universe_or_symbols(registry) -> None:
    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240110",
        },
        ToolContext(run_id="backtest-universe-required"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "UNIVERSE_UNSPECIFIED"
    assert result["symbols_source"] == "none"
    assert result["suggested_next_tools"] == [
        "create_universe_spec",
        "build_universe",
        "query_universe",
    ]


def test_backtest_returns_effective_universe_and_symbol_source(registry, lake: DataLake) -> None:
    _write_bars(lake)

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240215",
            "symbols": ["000001.SZ", "000002.SZ"],
            "top_n": 1,
        },
        ToolContext(run_id="backtest-universe-evidence"),
    )

    assert result["status"] == "completed"
    assert result["symbols_source"] == "explicit_symbols"
    assert result["symbols_count"] == 2
    assert result["symbols_sample"] == ["000001.SZ", "000002.SZ"]
    assert result["universe_effective"] in {"explicit_symbols", "stock_etf"}
    assert result["cost_estimate"]["estimated_symbols"] == 2


def test_backtest_blocks_broad_universe_with_three_symbols(registry, lake: DataLake) -> None:
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ", "000003.SZ"])

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240215",
            "universe_type": "stock",
            "top_n": 1,
        },
        ToolContext(run_id="backtest-broad-universe-too-small"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "BROAD_UNIVERSE_TOO_SMALL"
    assert result["universe_diagnostics"]["selected_count"] == 3
    assert result["universe_diagnostics"]["evidence_threshold"] == 500
    assert result["next_repair_action"] == "repair_universe_resolution_or_market_data_coverage"


def test_backtest_allows_explicit_small_symbol_list_when_user_requested_it(
    registry,
    lake: DataLake,
) -> None:
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ", "000003.SZ"])

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240215",
            "symbols": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "top_n": 1,
        },
        ToolContext(run_id="backtest-explicit-small-list"),
    )

    assert result["status"] == "completed"
    assert result["symbols_source"] == "explicit_symbols"
    assert result["symbols_count"] == 3


def _write_bars(lake: DataLake, *, symbols: list[str] | None = None) -> None:
    symbols = symbols or ["000001.SZ", "000002.SZ"]
    rows = []
    start = date(2024, 1, 1)
    for offset in range(46):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        for symbol_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "open": 10 + symbol_index + offset * 0.1,
                    "high": 11 + symbol_index + offset * 0.1,
                    "low": 9 + symbol_index + offset * 0.1,
                    "close": 10.5 + symbol_index + offset * 0.2,
                    "vol": 100000,
                    "amount": 1000000,
                }
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
