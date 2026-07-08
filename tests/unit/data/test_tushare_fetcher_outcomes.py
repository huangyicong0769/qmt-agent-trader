from __future__ import annotations

from typing import Any

import pandas as pd

from qmt_agent_trader.data.providers.base import FetchItem
from qmt_agent_trader.data.providers.tushare.client import TushareClient
from qmt_agent_trader.data.providers.tushare.fetcher import TushareFetcher
from qmt_agent_trader.data.providers.tushare.planner import TushareFetchPlanner
from qmt_agent_trader.data.storage import DataLake


class FakeClient(TushareClient):
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        super().__init__(token="fake")
        self.frames = frames

    def query(
        self,
        api_name: str,
        params: dict[str, Any],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        _ = params, fields
        return self.frames.get(api_name, pd.DataFrame()).copy()


def test_fetcher_zero_rows_is_no_data_and_skips_raw_write(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan = TushareFetchPlanner().plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=["000001.SZ"],
                fields=["ts_code", "trade_date", "pe_ttm"],
                start_date="20260708",
                end_date="20260708",
            )
        ]
    )

    result = TushareFetcher(FakeClient({"daily_basic": pd.DataFrame()}), lake).run(
        plan,
        execute_plan=True,
    )

    assert result.status == "NO_DATA"
    assert result.domain_status == "NO_DATA"
    assert result.evidence_status == "INCOMPLETE"
    assert result.coverage_status == "NO_DATA"
    assert result.writes == []
    assert result.dataset_results[0]["status"] == "NO_DATA"
    assert result.dataset_results[0]["rows"] == 0
    assert result.dataset_results[0]["write_skipped"] is True
    assert not lake.dataset_path("raw", "tushare/daily_basic").exists()


def test_fetcher_partial_update_keeps_zero_row_dataset_separate(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan = TushareFetchPlanner().plan(
        [
            FetchItem(
                api_name="daily",
                symbols=["000001.SZ"],
                fields=["ts_code", "trade_date", "open", "high", "low", "close"],
                start_date="20260708",
                end_date="20260708",
            ),
            FetchItem(
                api_name="fund_daily",
                symbols=["159259.SZ"],
                fields=["ts_code", "trade_date", "open", "high", "low", "close"],
                start_date="20260708",
                end_date="20260708",
            ),
        ]
    )
    client = FakeClient(
        {
            "daily": pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260708",
                        "open": 1.0,
                        "high": 1.1,
                        "low": 0.9,
                        "close": 1.0,
                    }
                ]
            ),
            "fund_daily": pd.DataFrame(),
        }
    )

    result = TushareFetcher(client, lake).run(plan, execute_plan=True)

    assert result.status == "PARTIAL_UPDATE"
    assert result.domain_status == "PARTIAL"
    assert result.evidence_status == "INCOMPLETE"
    assert result.coverage_status == "PARTIAL_COVERAGE"
    assert [write["dataset_id"] for write in result.writes] == ["tushare.daily"]
    assert {item["status"] for item in result.dataset_results} == {"updated", "NO_DATA"}
    assert "zero_rows_for_dataset:tushare.fund_daily" in result.warnings


def test_fetcher_schema_mismatch_is_invalid_without_raw_write(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan = TushareFetchPlanner().plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=["000001.SZ"],
                fields=["ts_code", "trade_date"],
                start_date="20260708",
                end_date="20260708",
            )
        ]
    )

    result = TushareFetcher(
        FakeClient({"daily_basic": pd.DataFrame([{"ts_code": "000001.SZ"}])}),
        lake,
    ).run(plan, execute_plan=True)

    assert result.status == "SCHEMA_MISMATCH"
    assert result.domain_status == "FAILED"
    assert result.evidence_status == "INVALID"
    assert result.coverage_status == "INVALID_REQUEST"
    assert result.writes == []
    assert result.dataset_results[0]["status"] == "SCHEMA_MISMATCH"
    assert not lake.dataset_path("raw", "tushare/daily_basic").exists()


def test_fetcher_all_positive_rows_is_valid_update(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan = TushareFetchPlanner().plan(
        [
            FetchItem(
                api_name="daily",
                symbols=["000001.SZ"],
                fields=["ts_code", "trade_date", "open", "high", "low", "close"],
                start_date="20260708",
                end_date="20260708",
            )
        ]
    )

    result = TushareFetcher(
        FakeClient(
            {
                "daily": pd.DataFrame(
                    [
                        {
                            "ts_code": "000001.SZ",
                            "trade_date": "20260708",
                            "open": 1.0,
                            "high": 1.1,
                            "low": 0.9,
                            "close": 1.0,
                        }
                    ]
                )
            }
        ),
        lake,
    ).run(plan, execute_plan=True)

    assert result.status == "updated"
    assert result.domain_status == "OK"
    assert result.evidence_status == "VALID"
    assert result.coverage_status == "OK"
    assert result.dataset_results[0]["rows"] == 1
