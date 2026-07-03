from __future__ import annotations

from typing import Any

import pandas as pd

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools import remote_data_tools
from qmt_agent_trader.agent.tools.base import AgentTool
from qmt_agent_trader.agent.tools.remote_data_tools import build_remote_data_tools, wire
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.providers.tushare.client import TushareClient
from qmt_agent_trader.data.storage import DataLake


class ExplodingGenericClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="fake")

    def query(
        self,
        api_name: str,
        params: dict[str, Any],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        raise AssertionError(f"unexpected live request: {api_name} {params} {fields}")


class RecordingGenericClient(TushareClient):
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        super().__init__(token="fake")
        self.frames = frames
        self.calls: list[tuple[str, dict[str, Any], list[str] | None]] = []

    def query(
        self,
        api_name: str,
        params: dict[str, Any],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        self.calls.append((api_name, params, fields))
        return self.frames.get(api_name, pd.DataFrame()).copy()


def _deps(tmp_path, lake: DataLake) -> AgentToolDependencies:
    return AgentToolDependencies(
        settings=Settings(project_root=tmp_path, tushare_token=None),
        data_lake=lake,
        sandbox=CodeSandbox(tmp_path / "generated"),
        experiment_store=ExperimentStore(tmp_path / "experiments"),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )


def _tools(tmp_path, lake: DataLake) -> dict[str, AgentTool]:
    return {item.spec.name: item for item in build_remote_data_tools(_deps(tmp_path, lake))}


def test_build_remote_data_tools_exposes_only_registry_surface(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")

    tools = _tools(tmp_path, lake)

    assert list(tools) == [
        "list_tushare_capabilities",
        "plan_tushare_fetch",
        "run_tushare_fetch",
        "build_data_table",
    ]
    assert not hasattr(remote_data_tools, "run_remote_data_update_tool")
    assert not hasattr(remote_data_tools, "run_fundamental_data_update_tool")
    assert not hasattr(remote_data_tools, "run_macro_data_update_tool")


def test_plan_tushare_fetch_rejects_unknown_fields_and_placeholders(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan_tool = _tools(tmp_path, lake)["plan_tushare_fetch"]

    unknown_field = plan_tool.run(
        {
            "items": [
                {
                    "api_name": "daily_basic",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "市盈率"],
                    "start_date": "20240101",
                    "end_date": "20240131",
                }
            ]
        },
        ToolContext(run_id="r-invalid-field"),
    )
    placeholder = plan_tool.run(
        {"items": [{"api_name": "repurchase"}]},
        ToolContext(run_id="r-placeholder"),
    )

    assert unknown_field["status"] == "INVALID_REQUEST"
    assert unknown_field["reason"] == "unknown_fields"
    assert unknown_field["errors"][0]["unknown_fields"] == ["市盈率"]
    assert placeholder["status"] == "NOT_IMPLEMENTED"
    assert placeholder["reason"] == "endpoint_registered_as_placeholder"


def test_run_tushare_fetch_dry_run_does_not_query_remote(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = ExplodingGenericClient()
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: client,
    )
    run_tool = _tools(tmp_path, lake)["run_tushare_fetch"]

    result = run_tool.run(
        {
            "items": [
                {
                    "api_name": "daily_basic",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "trade_date", "pe_ttm"],
                    "start_date": "20240101",
                    "end_date": "20240131",
                }
            ],
            "dry_run": True,
        },
        ToolContext(run_id="r-fetch-dry-run", dry_run=True),
    )

    assert result["status"] == "planned"
    assert result["dry_run"] is True
    assert result["execute_plan"] is False
    assert not lake.dataset_path("raw", "tushare/daily_basic").exists()


def test_run_tushare_fetch_live_writes_new_layout_and_metadata(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = RecordingGenericClient(
        {
            "daily_basic": pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20240102",
                        "pe_ttm": 5.0,
                    }
                ]
            )
        }
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: client,
    )
    run_tool = _tools(tmp_path, lake)["run_tushare_fetch"]

    result = run_tool.run(
        {
            "items": [
                {
                    "api_name": "daily_basic",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "trade_date", "pe_ttm"],
                    "start_date": "20240101",
                    "end_date": "20240131",
                }
            ],
            "execute_plan": True,
        },
        ToolContext(run_id="r-fetch-live", dry_run=False),
    )

    assert result["status"] == "updated"
    assert client.calls == [
        (
            "daily_basic",
            {"start_date": "20240101", "end_date": "20240131", "ts_code": "000001.SZ"},
            ["ts_code", "trade_date", "pe_ttm"],
        )
    ]
    assert lake.dataset_path("raw", "tushare/daily_basic").exists()
    assert not lake.dataset_path("raw", "tushare_daily_basic").exists()
    state = lake.query_parquet("SELECT * FROM data_fetch_state_v2").to_dict(orient="records")
    assert state[0]["dataset_id"] == "tushare.daily_basic"
    assert state[0]["coverage_start"] == "20240101"
    assert state[0]["coverage_end"] == "20240131"


def test_large_marketwide_fetch_requires_new_layout_trade_calendar(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan_tool = _tools(tmp_path, lake)["plan_tushare_fetch"]
    request = {
        "items": [
            {
                "api_name": "daily_basic",
                "symbols": [f"{index:06d}.SZ" for index in range(1, 40)],
                "fields": ["ts_code", "trade_date", "pe_ttm"],
                "start_date": "20240101",
                "end_date": "20240105",
            }
        ]
    }

    missing_calendar = plan_tool.run(request, ToolContext(run_id="r-no-calendar"))
    lake.write_parquet(
        pd.DataFrame(
            [{"cal_date": "20240102", "is_open": 1}, {"cal_date": "20240103", "is_open": 1}]
        ),
        "raw",
        "tushare_trade_calendar",
    )
    legacy_calendar = plan_tool.run(request, ToolContext(run_id="r-legacy-calendar"))
    lake.write_parquet(
        pd.DataFrame(
            [{"cal_date": "20240102", "is_open": 1}, {"cal_date": "20240103", "is_open": 1}]
        ),
        "raw",
        "tushare/trade_cal",
    )
    new_calendar = plan_tool.run(request, ToolContext(run_id="r-new-calendar"))

    assert missing_calendar["status"] == "BLOCKED"
    assert missing_calendar["reason"] == "TRADE_CALENDAR_REQUIRED"
    assert legacy_calendar["status"] == "BLOCKED"
    assert legacy_calendar["reason"] == "TRADE_CALENDAR_REQUIRED"
    assert new_calendar["status"] == "planned"
    assert new_calendar["strategy"] == "marketwide_by_trade_date"
    assert [batch["params"]["trade_date"] for batch in new_calendar["batches"]] == [
        "20240102",
        "20240103",
    ]


def test_build_data_table_rejects_research_daily_wide(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    build_tool = _tools(tmp_path, lake)["build_data_table"]

    result = build_tool.run(
        {"table": "research_daily_wide"},
        ToolContext(run_id="r-reject-wide-table"),
    )

    assert result["status"] == "INVALID_REQUEST"
    assert "research_daily_wide" not in result["allowed_tables"]
