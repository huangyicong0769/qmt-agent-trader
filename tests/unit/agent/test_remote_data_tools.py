from __future__ import annotations

import pandas as pd
from pydantic import SecretStr

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.remote_data_tools import (
    plan_remote_data_update_tool,
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


class FailingClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        raise RuntimeError("upstream rejected request with secret-token")


def test_plan_remote_data_update_is_read_only(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path),
        client_factory=lambda: ExplodingClient(),
    )

    result = plan_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
        ToolContext(run_id="r-plan", dry_run=True),
    )

    assert result["status"] == "planned"
    assert result["source"] == "tushare"
    assert result["missing_ranges"] == [{"start_date": "20240101", "end_date": "20240103"}]
    assert not lake.dataset_path("raw", "tushare_daily").exists()


def test_run_remote_data_update_dry_run_does_not_require_token_or_fetch(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
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
