from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.query_tools import query_fundamentals_pit_tool, set_data_lake
from qmt_agent_trader.data.storage import DataLake


def test_query_fundamentals_pit_returns_not_available_without_lake() -> None:
    set_data_lake(None)  # type: ignore[arg-type]

    result = query_fundamentals_pit_tool.run(
        {"symbol": "000001.SZ", "as_of_date": "20240131"},
        ToolContext(run_id="fundamentals-no-lake"),
    )

    assert result["metadata"]["status"] == "NOT_AVAILABLE"


def test_query_fundamentals_pit_returns_no_data_when_raw_missing(tmp_path) -> None:
    set_data_lake(DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb"))

    result = query_fundamentals_pit_tool.run(
        {"symbol": "000001.SZ", "as_of_date": "20240131"},
        ToolContext(run_id="fundamentals-no-data"),
    )

    assert result["rows"] == []
    assert result["metadata"]["status"] == "NO_DATA"
    assert result["metadata"]["coverage_status"] == "NO_DATA"
    assert result["metadata"]["missing_ranges"] == [
        {"start_date": "20240131", "end_date": "20240131"}
    ]
    assert result["metadata"]["next_repair_tool"] == "run_tushare_fetch"


def test_query_fundamentals_pit_returns_daily_and_financial_fields(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240131",
                    "pe_ttm": 4.8,
                    "pb": 0.55,
                    "total_mv": 1000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily_basic",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20230930",
                    "ann_date": "20231025",
                    "roe": 0.11,
                    "gross_margin": 31.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240201",
                    "roe": 0.12,
                    "gross_margin": 32.0,
                },
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )
    set_data_lake(lake)

    result = query_fundamentals_pit_tool.run(
        {
            "symbols": ["000001.SZ"],
            "as_of_date": "20240131",
            "fields": ["pe_ttm", "pb", "roe", "gross_margin", "total_mv"],
        },
        ToolContext(run_id="fundamentals-ok"),
    )

    assert result["metadata"]["status"] == "OK"
    assert result["metadata"]["point_in_time"] is True
    assert result["metadata"]["datasets_used"] == [
        "tushare/daily_basic",
        "tushare/fina_indicator",
    ]
    assert result["rows"] == [
        {
            "symbol": "000001.SZ",
            "trade_date": "2024-01-31",
            "pe_ttm": 4.8,
            "pb": 0.55,
            "total_mv": 1000.0,
            "as_of_date": "2024-01-31",
            "data_status": "OK",
            "latest_period_end": "2023-09-30",
            "latest_announced_at": "2023-10-25",
            "roe": 0.11,
            "gross_margin": 31.0,
        }
    ]


def test_query_fundamentals_pit_reports_partial_coverage(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240131", "pe_ttm": 4.8}]),
        "raw",
        "tushare/daily_basic",
    )
    set_data_lake(lake)

    result = query_fundamentals_pit_tool.run(
        {
            "symbols": ["000001.SZ", "000002.SZ"],
            "as_of_date": "20240131",
            "fields": ["pe_ttm", "roe"],
        },
        ToolContext(run_id="fundamentals-partial"),
    )

    assert result["metadata"]["status"] == "PARTIAL_COVERAGE"
    assert result["metadata"]["missing_symbols"] == ["000002.SZ"]
    assert result["metadata"]["missing_fields"] == {"roe": ["000001.SZ"]}
    assert result["repair_action"]["fetch_items"][0]["api_name"] == "daily_basic"
    assert result["repair_action"]["fetch_items"][1]["api_name"] == "fina_indicator"


def test_query_fundamentals_pit_repairs_missing_daily_basic_fields(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "end_date": "20231231", "ann_date": "20240110"}]),
        "raw",
        "tushare/fina_indicator",
    )
    set_data_lake(lake)

    result = query_fundamentals_pit_tool.run(
        {
            "symbols": ["000001.SZ"],
            "as_of_date": "20240131",
            "fields": ["pe_ttm", "pb", "total_mv"],
        },
        ToolContext(run_id="fundamentals-daily-basic-repair"),
    )

    assert result["status"] == "PARTIAL_COVERAGE"
    assert result["repair_action"]["reason"] == "missing_daily_basic_coverage"
    assert result["repair_action"]["fetch_items"] == [
        {
            "api_name": "daily_basic",
            "symbols": ["000001.SZ"],
            "fields": ["ts_code", "trade_date", "pb", "pe_ttm", "total_mv"],
            "start_date": "20240131",
            "end_date": "20240131",
        }
    ]


def test_query_fundamentals_pit_repairs_roe_with_fina_indicator(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240131", "pe_ttm": 4.8}]),
        "raw",
        "tushare/daily_basic",
    )
    set_data_lake(lake)

    result = query_fundamentals_pit_tool.run(
        {
            "symbols": ["000001.SZ"],
            "as_of_date": "20240131",
            "fields": ["roe"],
        },
        ToolContext(run_id="fundamentals-roe-repair"),
    )

    assert result["repair_action"]["fetch_items"][0]["api_name"] == "fina_indicator"


def test_query_fundamentals_pit_repairs_total_revenue_with_income(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240131", "pe_ttm": 4.8}]),
        "raw",
        "tushare/daily_basic",
    )
    set_data_lake(lake)

    result = query_fundamentals_pit_tool.run(
        {
            "symbols": ["000001.SZ"],
            "as_of_date": "20240131",
            "fields": ["total_revenue"],
        },
        ToolContext(run_id="fundamentals-income-repair"),
    )

    assert result["repair_action"]["fetch_items"][0]["api_name"] == "income"


def test_query_fundamentals_pit_unknown_field_requires_capability_discovery(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240131", "pe_ttm": 4.8}]),
        "raw",
        "tushare/daily_basic",
    )
    set_data_lake(lake)

    result = query_fundamentals_pit_tool.run(
        {
            "symbols": ["000001.SZ"],
            "as_of_date": "20240131",
            "fields": ["mystery_ratio"],
        },
        ToolContext(run_id="fundamentals-unknown-repair"),
    )

    assert result["next_repair_tool"] == "list_tushare_capabilities"
    assert result["repair_action"]["type"] == "capability_discovery_required"
    assert result["repair_action"]["tool"] == "list_tushare_capabilities"
