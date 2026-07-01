from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest
from qmt_agent_trader.services.data_update_service import (
    TushareDataUpdateService,
    build_fundamental_update_plan,
)


class FakeFundamentalClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="fake")
        self.requests: list[TushareRequest] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        self.requests.append(request)
        if request.api_name == "trade_cal":
            return pd.DataFrame(
                [
                    {"cal_date": "20240102", "is_open": 1},
                    {"cal_date": "20240103", "is_open": 1},
                ]
            )
        if request.api_name == "daily_basic":
            trade_date = request.params.get("trade_date", request.params.get("start_date"))
            return pd.DataFrame(
                [
                    {
                        "ts_code": request.params.get("ts_code", "000001.SZ"),
                        "trade_date": trade_date,
                        "pe_ttm": 4.8,
                        "pb": 0.55,
                        "dv_ttm": 3.0,
                        "total_mv": 1000.0,
                        "circ_mv": 900.0,
                    }
                ]
            )
        if request.api_name == "income":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20231231",
                        "ann_date": "20240120",
                        "report_type": "1",
                        "n_income_attr_p": 100.0,
                    }
                ]
            )
        if request.api_name == "balancesheet":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20231231",
                        "ann_date": "20240120",
                        "report_type": "1",
                        "total_assets": 1000.0,
                    }
                ]
            )
        if request.api_name == "cashflow":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20231231",
                        "ann_date": "20240120",
                        "report_type": "1",
                        "n_cashflow_act": 10.0,
                    }
                ]
            )
        if request.api_name == "fina_indicator":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20231231",
                        "ann_date": "20240120",
                        "roe": 0.11,
                        "gross_margin": 32.0,
                    }
                ]
            )
        if request.api_name == "dividend":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20231231",
                        "ann_date": "20240120",
                        "div_proc": "实施",
                        "cash_div_tax": 1.2,
                    }
                ]
            )
        raise AssertionError(request.api_name)


def test_build_fundamental_update_plan_contains_targets() -> None:
    plan = build_fundamental_update_plan(TushareClient(token="fake"), "2024-01-01", "2024-01-31")

    targets = {item["target_dataset"] for item in plan}
    assert "tushare_daily_basic" in targets
    assert "tushare_income" in targets
    assert "tushare_fina_indicator" in targets
    daily_basic = next(item for item in plan if item["target_dataset"] == "tushare_daily_basic")
    assert daily_basic["api_name"] == "daily_basic"
    assert daily_basic["incremental_key_columns"] == ["ts_code", "trade_date"]
    assert daily_basic["pit_safe"] is True


def test_update_fundamentals_fetches_daily_basic_by_missing_open_dates(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102"}]),
        "raw",
        "tushare_daily_basic",
    )
    client = FakeFundamentalClient()

    result = TushareDataUpdateService(client, lake).update_fundamentals(
        "20240102",
        "20240103",
        include_financial_statements=False,
    )

    daily_basic_requests = [
        request for request in client.requests if request.api_name == "daily_basic"
    ]
    assert [request.params for request in daily_basic_requests] == [{"trade_date": "20240103"}]
    assert result.open_dates == ["20240102", "20240103"]
    assert lake.read_parquet("raw", "tushare_daily_basic")["trade_date"].tolist() == [
        "20240102",
        "20240103",
    ]


def test_update_fundamentals_writes_financial_tables_incrementally(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeFundamentalClient()
    service = TushareDataUpdateService(client, lake)

    service.update_fundamentals("20240101", "20240131")
    service.update_fundamentals("20240101", "20240131")

    fina = lake.read_parquet("raw", "tushare_fina_indicator")
    assert len(fina) == 1
    assert fina["roe"].tolist() == [0.11]
    assert lake.fetch_state("tushare", "tushare_income")[0]["status"] == "success"


def test_update_fundamentals_supports_scoped_ts_code_daily_basic(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeFundamentalClient()

    TushareDataUpdateService(client, lake).update_fundamentals(
        "20240101",
        "20240131",
        ts_code="000001.SZ",
        include_financial_statements=False,
    )

    daily_basic_request = next(
        request for request in client.requests if request.api_name == "daily_basic"
    )
    assert daily_basic_request.params == {
        "start_date": "20240101",
        "end_date": "20240131",
        "ts_code": "000001.SZ",
    }
