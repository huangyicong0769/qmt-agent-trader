from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry, strategy_tools
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.resolver import UniverseResolver


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


def test_rolling_backtest_uses_per_date_resolved_symbol_sets(
    registry,
    lake: DataLake,
) -> None:
    _write_backtest_bars(lake)
    _write_stock_basic(lake, listed_c_date="20240125")

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240210",
            "universe_mode": "rolling",
            "universe_spec": _stock_universe_spec(mode="rolling"),
            "top_n": 1,
        },
        ToolContext(run_id="rolling-backtest"),
    )

    assert result["status"] == "completed"
    assert result["universe_mode"] == "rolling"
    assert result["symbols_source"] == "universe_rolling"
    assert result["rolling_universe_stats"]["changed_dates"] > 0
    assert result["rolling_universe_stats"]["empty_dates"] == []
    assert "20240124" in result["universe_resolution"]["rolling_symbols"]
    assert "20240125" in result["universe_resolution"]["rolling_symbols"]
    assert result["universe_resolution"]["rolling_symbols"]["20240124"] == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert result["universe_resolution"]["rolling_symbols"]["20240125"] == [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    ]

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    trades = report["payload"]["trades"]
    early_c_trades = [
        trade
        for trade in trades
        if trade["symbol"] == "000003.SZ" and trade["signal_date"] < "2024-01-25"
    ]
    later_c_trades = [
        trade
        for trade in trades
        if trade["symbol"] == "000003.SZ" and trade["signal_date"] >= "2024-01-25"
    ]
    assert early_c_trades == []
    assert later_c_trades


def test_backtest_blocks_on_empty_rolling_universe(registry, lake: DataLake) -> None:
    rows = [
        _bar("000001.SZ", "20240101", 10.0, st=True),
        _bar("000001.SZ", "20240102", 10.1, st=True),
    ]
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
    _write_trade_state_sources(lake, rows)
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
    _write_stock_basic(lake, listed_c_date="20200101")

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "start_date": "20240101",
            "end_date": "20240102",
            "universe_mode": "rolling",
            "universe_spec": _stock_universe_spec(mode="rolling"),
        },
        ToolContext(run_id="rolling-empty"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "ROLLING_UNIVERSE_EMPTY"
    assert result["empty_dates"] == ["20240101", "20240102"]
    assert result["suggested_next_tools"] == ["inspect_universe", "build_universe", "query_bars"]


def test_universe_frequency_is_explicitly_separate_from_strategy_frequency(
    monkeypatch,
    lake: DataLake,
) -> None:
    observed: dict[str, object] = {}

    def fake_build(self, spec, **kwargs):
        observed["frequency"] = kwargs["rebalance_frequency"]
        return {
            "status": "OK",
            "rolling_symbols": {"20240102": ["000001.SZ"]},
            "metadata": {"resolve_dates": ["20240102"], "empty_dates": []},
        }

    monkeypatch.setattr(UniverseResolver, "build", fake_build)
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)

    strategy_tools._run_backtest(
        {
            "strategy_spec": {
                "strategy_id": "weekly_value",
                "name": "Weekly value",
                "kind": "FACTOR_RANK_LONG_ONLY",
                "factors": [{"factor_id": "pb_rank", "ascending": True}],
                "portfolio": {"top_n": 10},
                "rebalance": {"frequency": "weekly"},
            },
            "universe_spec": _stock_universe_spec(mode="rolling"),
            "universe_mode": "rolling",
            "universe_rebalance_frequency": "monthly",
            "start_date": "20240101",
            "end_date": "20240331",
        },
        ToolContext(run_id="separate-universe-frequency"),
    )

    assert observed["frequency"] == "monthly"


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
    start = date(2023, 12, 12)
    for offset in range(61):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        rows.extend(
            [
                _bar("000001.SZ", trade_date, 10.0 + offset * 0.05),
                _bar("000002.SZ", trade_date, 12.0 + offset * 0.03),
                _bar("000003.SZ", trade_date, 8.0 + offset * 0.30),
            ]
        )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
    _write_trade_state_sources(lake, rows)
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


def _write_trade_state_sources(
    lake: DataLake,
    rows: list[dict[str, object]],
) -> None:
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": row["ts_code"],
                    "trade_date": row["trade_date"],
                    "up_limit": float(row["close"]) * 1.1,
                    "down_limit": float(row["close"]) * 0.9,
                }
                for row in rows
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    st_rows = [row for row in rows if bool(row.get("st", False))]
    st_periods = [
        {
            "ts_code": symbol,
            "name": "ST test",
            "start_date": min(
                str(row["trade_date"])
                for row in st_rows
                if row["ts_code"] == symbol
            ),
            "end_date": max(
                str(row["trade_date"])
                for row in st_rows
                if row["ts_code"] == symbol
            ),
        }
        for symbol in sorted({str(row["ts_code"]) for row in st_rows})
    ]
    lake.write_parquet(
        pd.DataFrame(
            st_periods,
            columns=["ts_code", "name", "start_date", "end_date"],
        ),
        "raw",
        "tushare/namechange",
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


def _bar(
    symbol: str,
    trade_date: str,
    close: float,
    *,
    st: bool = False,
) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "trade_date": trade_date,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "vol": 1000.0,
        "amount": close * 1000,
        "st": st,
    }


def _stock_basic(symbol: str, name: str, list_date: str) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "name": name,
        "industry": "软件服务",
        "list_status": "L",
        "list_date": list_date,
    }
