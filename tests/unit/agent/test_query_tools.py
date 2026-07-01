from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.query_tools import (
    list_data_catalog_tool,
    query_bars_tool,
    query_universe_tool,
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


def test_query_bars_reports_partial_coverage_for_multi_symbol_request(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260629",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vol": 200,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20260629",
                    "open": 20.0,
                    "high": 21.0,
                    "low": 19.0,
                    "close": 20.5,
                    "vol": 300,
                },
            ]
        ),
        "raw",
        "tushare_daily",
    )
    set_data_lake(lake)

    result = query_bars_tool.run(
        {
            "symbols": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "start_date": "20260629",
            "end_date": "20260629",
        },
        ToolContext(run_id="bars-partial"),
    )

    assert len(result["rows"]) == 2
    metadata = result["metadata"]
    assert metadata["status"] == "PARTIAL_COVERAGE"
    assert metadata["covered_symbols"] == ["000001.SZ", "000002.SZ"]
    assert metadata["missing_symbols"] == ["000003.SZ"]
    assert metadata["stale_symbols"] == []
    assert metadata["coverage_by_symbol"]["000001.SZ"]["returned"] == 1
    assert metadata["coverage_by_symbol"]["000003.SZ"]["returned"] == 0


def test_query_bars_reports_no_matching_bars_for_requested_symbols(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260629",
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
        {
            "symbols": ["000003.SZ", "000004.SZ"],
            "start_date": "20260629",
            "end_date": "20260629",
        },
        ToolContext(run_id="bars-none"),
    )

    assert result["rows"] == []
    metadata = result["metadata"]
    assert metadata["status"] == "NO_MATCHING_BARS"
    assert metadata["missing_symbols"] == ["000003.SZ", "000004.SZ"]
    assert metadata["covered_symbols"] == []
    assert metadata["stale_symbols"] == []


def test_query_bars_reports_stale_symbols_when_end_is_not_covered(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260626",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vol": 200,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20260629",
                    "open": 20.0,
                    "high": 21.0,
                    "low": 19.0,
                    "close": 20.5,
                    "vol": 300,
                },
            ]
        ),
        "raw",
        "tushare_daily",
    )
    set_data_lake(lake)

    result = query_bars_tool.run(
        {
            "symbols": ["000001.SZ", "000002.SZ"],
            "start_date": "20260626",
            "end_date": "20260629",
        },
        ToolContext(run_id="bars-stale"),
    )

    metadata = result["metadata"]
    assert metadata["status"] == "PARTIAL_COVERAGE"
    assert metadata["covered_symbols"] == ["000002.SZ"]
    assert metadata["stale_symbols"] == ["000001.SZ"]
    assert metadata["missing_symbols"] == []
    assert metadata["coverage_by_symbol"]["000001.SZ"]["data_freshness"] == (
        "stale_vs_requested_end"
    )


def test_query_bars_pushes_limit_and_can_skip_trade_state(tmp_path, monkeypatch) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    set_data_lake(lake)
    calls: list[dict[str, object]] = []

    def fake_limited(lake_arg, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": pd.Timestamp("2024-01-02").date(),
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.2,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "turnover": 1.0,
                    "suspended": False,
                    "limit_up": False,
                    "limit_down": False,
                    "st": False,
                }
            ]
        )

    monkeypatch.setattr(
        "qmt_agent_trader.agent.tools.query_tools._load_bars_for_query",
        fake_limited,
    )

    result = query_bars_tool.run(
        {
            "start_date": "20200101",
            "end_date": "20250101",
            "limit": 1,
            "include_trade_state": False,
        },
        ToolContext(run_id="bars-limit"),
    )

    assert result["metadata"]["limit"] == 1
    assert result["metadata"]["backend_limited"] is True
    assert result["metadata"]["include_trade_state"] is False
    assert calls[0]["limit"] == 1
    assert calls[0]["include_trade_state"] is False


def test_query_bars_rejects_limit_above_maximum(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    set_data_lake(lake)

    result = query_bars_tool.run({"limit": 10001}, ToolContext(run_id="bars-bad-limit"))

    assert result["metadata"]["status"] == "INVALID_REQUEST"


def test_query_universe_builds_reproducible_cyclical_basket_from_stock_basic(
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "600019.SH",
                    "trade_date": "20240628",
                    "open": 5.0,
                    "high": 5.2,
                    "low": 4.9,
                    "close": 5.1,
                },
                {
                    "ts_code": "600036.SH",
                    "trade_date": "20240628",
                    "open": 30.0,
                    "high": 31.0,
                    "low": 29.0,
                    "close": 30.5,
                },
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20240628",
                    "open": 1500.0,
                    "high": 1510.0,
                    "low": 1490.0,
                    "close": 1505.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240628",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                },
            ]
        ),
        "raw",
        "tushare_daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "600019.SH",
                    "name": "宝钢股份",
                    "industry": "钢铁",
                    "list_date": "20001212",
                    "list_status": "L",
                },
                {
                    "ts_code": "600036.SH",
                    "name": "招商银行",
                    "industry": "银行",
                    "list_date": "20020409",
                    "list_status": "L",
                },
                {
                    "ts_code": "600519.SH",
                    "name": "贵州茅台",
                    "industry": "白酒",
                    "list_date": "20010827",
                    "list_status": "L",
                },
                {
                    "ts_code": "000001.SZ",
                    "name": "*ST平安",
                    "industry": "银行",
                    "list_date": "19910403",
                    "list_status": "L",
                },
                {
                    "ts_code": "600999.SH",
                    "name": "无行情周期股",
                    "industry": "煤炭",
                    "list_date": "20000101",
                    "list_status": "L",
                },
            ]
        ),
        "raw",
        "tushare_stock_basic",
    )
    set_data_lake(lake)

    result = query_universe_tool.run(
        {
            "as_of_date": "20240628",
            "filters": {"theme": "cyclical", "min_listed_days": 60},
        },
        ToolContext(run_id="cyclical-universe"),
    )

    assert result["status"] == "OK"
    assert result["symbols"] == ["600019.SH", "600036.SH"]
    assert result["metadata"]["theme"] == "cyclical"
    assert result["metadata"]["selection_rules"]["industry_source"] == "tushare_stock_basic"
    assert result["metadata"]["industry_distribution"] == {"钢铁": 1, "银行": 1}
    excluded = {item["symbol"]: item["reason"] for item in result["metadata"]["excluded_symbols"]}
    assert excluded["600519.SH"] == "industry_not_in_theme"
    assert excluded["000001.SZ"] == "st"
    assert excluded["600999.SH"] == "no_bar_coverage"


def test_query_universe_defaults_to_current_date_for_cyclical_theme(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "600036.SH",
                    "trade_date": "20240628",
                    "open": 30.0,
                    "high": 31.0,
                    "low": 29.0,
                    "close": 30.5,
                }
            ]
        ),
        "raw",
        "tushare_daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "600036.SH",
                    "name": "招商银行",
                    "industry": "银行",
                    "list_date": "20020409",
                    "list_status": "L",
                }
            ]
        ),
        "raw",
        "tushare_stock_basic",
    )
    set_data_lake(lake)

    result = query_universe_tool.run(
        {"filters": {"theme": "cyclical", "min_listed_days": 60}},
        ToolContext(run_id="cyclical-universe-default-date"),
    )

    assert result["status"] == "OK"
    assert result["symbols"] == ["600036.SH"]
    assert result["metadata"]["as_of_date"] == "2024-06-28"
