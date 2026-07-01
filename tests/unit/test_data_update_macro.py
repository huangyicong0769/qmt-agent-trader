from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest
from qmt_agent_trader.services.data_update_service import (
    TushareDataUpdateService,
    build_macro_update_plan,
)


class FakeMacroClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="fake")
        self.requests: list[TushareRequest] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        self.requests.append(request)
        if request.api_name == "shibor":
            return pd.DataFrame(
                [
                    {"date": "20240102", "on": 1.5, "1w": 1.8},
                    {"date": "20240103", "on": 1.6, "1w": 1.9},
                ]
            )
        if request.api_name == "cn_cpi":
            raise RuntimeError("permission denied")
        raise AssertionError(request.api_name)


def test_build_macro_update_plan_reports_dataset_metadata() -> None:
    plan = build_macro_update_plan(
        TushareClient(token="fake"),
        "2024-01-01",
        "2024-01-31",
        datasets=["shibor", "unknown"],
    )

    assert plan[0]["dataset"] == "shibor"
    assert plan[0]["target_dataset"] == "tushare_macro_shibor"
    assert plan[0]["incremental_key_columns"] == ["date"]
    assert plan[0]["pit_safe"] is True
    assert plan[1]["status"] == "INVALID_REQUEST"


def test_update_macro_writes_available_dataset_and_records_errors(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeMacroClient()

    result = TushareDataUpdateService(client, lake, retry_attempts=1).update_macro(
        "20240101",
        "20240131",
        datasets=["shibor", "cn_cpi", "unknown"],
    )

    assert [request.api_name for request in client.requests] == ["shibor", "cn_cpi"]
    assert lake.dataset_path("raw", "tushare_macro_shibor").exists()
    assert lake.read_parquet("raw", "tushare_macro_shibor")["date"].tolist() == [
        "20240102",
        "20240103",
    ]
    assert result.metadata is not None
    assert result.metadata["errors"] == {
        "cn_cpi": "permission denied",
        "unknown": "unknown macro dataset",
    }
    assert lake.fetch_state("tushare", "tushare_macro_cn_cpi")[0]["status"] == "error"
