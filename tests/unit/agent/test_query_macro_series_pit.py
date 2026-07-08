from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.query_tools import query_macro_series_pit_tool, set_data_lake
from qmt_agent_trader.data.storage import DataLake


def test_query_macro_series_pit_rejects_unknown_dataset_with_argument_repair(tmp_path) -> None:
    set_data_lake(DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb"))

    result = query_macro_series_pit_tool.run(
        {"dataset": "cpi", "as_of_date": "20240131"},
        ToolContext(run_id="macro-invalid"),
    )

    assert result["metadata"]["status"] == "INVALID_REQUEST"
    assert result["domain_status"] == "INVALID_REQUEST"
    assert result["evidence_status"] == "INVALID"
    assert result["coverage_status"] == "INVALID_REQUEST"
    assert result["metadata"]["next_repair_tool"] == "list_tushare_capabilities"
    assert result["next_repair_tool"] == "list_tushare_capabilities"
    assert result["repair_action"]["type"] == "fix_request_argument"
    assert result["repair_action"]["suggested_dataset"] == "cn_cpi"
    assert "cn_cpi" in result["metadata"]["known_datasets"]


def test_query_macro_series_pit_returns_no_data_for_missing_dataset(tmp_path) -> None:
    set_data_lake(DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb"))

    result = query_macro_series_pit_tool.run(
        {
            "dataset": "cn_cpi",
            "as_of_date": "20240131",
            "start_date": "20240101",
            "end_date": "20240131",
        },
        ToolContext(run_id="macro-no-data"),
    )

    assert result["rows"] == []
    assert result["metadata"]["status"] == "NO_DATA"
    assert result["metadata"]["coverage_status"] == "NO_DATA"
    assert result["metadata"]["missing_ranges"] == [
        {"start_date": "20240101", "end_date": "20240131"}
    ]
    assert result["metadata"]["next_repair_tool"] == "run_tushare_fetch"
    assert result["repair_action"]["fetch_items"][0]["api_name"] == "cn_cpi"


def test_query_macro_series_pit_returns_pit_safe_series(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"date": "20240102", "on": 1.5, "1w": 1.8},
                {"date": "20240201", "on": 1.6, "1w": 1.9},
            ]
        ),
        "raw",
        "tushare/shibor",
    )
    set_data_lake(lake)

    result = query_macro_series_pit_tool.run(
        {
            "dataset": "shibor",
            "as_of_date": "20240131",
            "start_date": "20240101",
            "fields": ["on"],
            "strict_pit": True,
        },
        ToolContext(run_id="macro-ok"),
    )

    assert result["metadata"]["status"] == "OK"
    assert result["metadata"]["visibility_rule"] == "date same-day visibility"
    assert result["metadata"]["actual_start"] == "2024-01-02"
    assert result["metadata"]["actual_end"] == "2024-01-02"
    assert result["rows"] == [
        {
            "date": "20240102",
            "period_date": "2024-01-02",
            "visible_date": "2024-01-02",
            "on": 1.5,
        }
    ]


def test_query_macro_series_pit_blocks_strict_unvalidated_dataset(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"month": "202401", "nt_val": 100.0}]),
        "raw",
        "tushare/cn_cpi",
    )
    set_data_lake(lake)

    result = query_macro_series_pit_tool.run(
        {
            "dataset": "cn_cpi",
            "as_of_date": "20240220",
            "strict_pit": True,
        },
        ToolContext(run_id="macro-unvalidated"),
    )

    assert result["rows"] == []
    assert result["metadata"]["status"] == "PIT_NOT_VALIDATED"
    assert result["coverage_status"] == "PARTIAL_COVERAGE"
    assert result["metadata"]["actual_start"] == "202401"
    assert result["metadata"]["actual_end"] == "202401"
    assert "warning" in result["metadata"]
