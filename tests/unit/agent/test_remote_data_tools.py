from __future__ import annotations

import pandas as pd
from pydantic import SecretStr

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.remote_data_tools import (
    run_remote_data_update_tool,
    wire,
)
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest


class ExplodingClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        raise AssertionError(f"unexpected live request: {request.api_name}")


class RecordingClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")
        self.seen: list[str] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        self.seen.append(request.api_name)
        if request.api_name == "trade_cal":
            return pd.DataFrame([{"cal_date": "20240102", "is_open": 1}])
        if request.api_name == "daily":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": request.params["trade_date"],
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                    }
                ]
            )
        if request.api_name == "suspend_d":
            return pd.DataFrame()
        if request.api_name == "stk_limit":
            return pd.DataFrame()
        raise AssertionError(f"unexpected request: {request.api_name}")


class FailingClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        raise RuntimeError("upstream rejected request with secret-token")


def test_run_remote_data_update_fetches_even_when_agent_context_is_dry_run(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = RecordingClient()
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path),
        client_factory=lambda: client,
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240102",
            "include_basics": False,
        },
        ToolContext(run_id="r-live-agent", dry_run=True),
    )

    assert result["status"] == "updated"
    assert "daily" in client.seen
    assert lake.dataset_path("raw", "tushare_daily").exists()


def test_run_remote_data_update_supports_explicit_dry_run_plan(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240101",
            "end_date": "20240103",
            "dry_run": True,
        },
        ToolContext(run_id="r-dry", dry_run=True),
    )

    assert result["status"] == "planned"
    assert "requests" in result
    assert not lake.dataset_path("raw", "tushare_daily").exists()


def test_run_remote_data_update_rejects_missing_token_for_live_fetch(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(data_lake=lake, settings=Settings(project_root=tmp_path, tushare_token=None))

    result = run_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
        ToolContext(run_id="r-live", dry_run=False),
    )

    assert result["status"] == "NOT_CONFIGURED"
    assert "token" in result["message"].lower()


def test_run_remote_data_update_rejects_oversized_ranges(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(
            project_root=tmp_path,
            tushare_token=SecretStr("secret-token"),
            remote_data_max_days_per_call=1,
        ),
    )

    result = run_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
        ToolContext(run_id="r-large", dry_run=False),
    )

    assert result["status"] == "INVALID_REQUEST"
    assert "remote_data_max_days_per_call" in result["message"]


def test_run_remote_data_update_sanitizes_service_errors(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(
            project_root=tmp_path,
            tushare_token=SecretStr("secret-token"),
        ),
        client_factory=lambda: FailingClient(),
    )

    result = run_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
        ToolContext(run_id="r-error", dry_run=False),
    )

    assert result["status"] == "error"
    assert "secret-token" not in result["message"]
