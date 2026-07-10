from __future__ import annotations

import json
from typing import Any

import pandas as pd

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools import remote_data_tools
from qmt_agent_trader.agent.tools.base import AgentTool
from qmt_agent_trader.agent.tools.remote_data_tools import build_remote_data_tools, wire
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.providers.tushare.client import TushareClient
from qmt_agent_trader.data.providers.tushare.ledger_migration import (
    repair_tushare_usage_ledger,
)
from qmt_agent_trader.data.providers.tushare.quota import TushareUsageLedger
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.initialization import initialize_persistence


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


def test_plan_tushare_fetch_reports_new_layout_local_coverage(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan_tool = _tools(tmp_path, lake)["plan_tushare_fetch"]
    request = {
        "items": [
            {
                "api_name": "daily_basic",
                "symbols": ["000001.SZ", "000002.SZ"],
                "fields": ["ts_code", "trade_date", "pe_ttm"],
                "start_date": "20240101",
                "end_date": "20240131",
            }
        ]
    }

    missing = plan_tool.run(request, ToolContext(run_id="r-coverage-missing"))
    lake.write_incremental_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "pe_ttm": 5.0,
                }
            ]
        ),
        "raw",
        "tushare/daily_basic",
        key_columns=["ts_code", "trade_date"],
    )
    partial = plan_tool.run(request, ToolContext(run_id="r-coverage-partial"))

    assert missing["status"] == "planned"
    assert missing["coverage_status"] == "NOT_VERIFIED"
    assert missing["local_coverage_status"] == "NO_DATA"
    assert missing["local_coverage"][0]["reason"] == "raw_dataset_missing"
    assert partial["status"] == "planned"
    assert partial["coverage_status"] == "NOT_VERIFIED"
    assert partial["local_coverage_status"] == "PARTIAL_COVERAGE"
    assert partial["local_coverage"][0]["missing_symbols"] == ["000002.SZ"]
    assert partial["local_coverage"][0]["partial_reasons"] == [
        "starts_after_requested_start",
        "ends_before_requested_end",
        "missing_symbols",
    ]


def test_plan_tushare_fetch_reports_missing_symbols_without_date_range(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_incremental_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "平安银行",
                    "list_date": "19910403",
                }
            ]
        ),
        "raw",
        "tushare/stock_basic",
        key_columns=["ts_code"],
    )
    plan_tool = _tools(tmp_path, lake)["plan_tushare_fetch"]

    result = plan_tool.run(
        {
            "items": [
                {
                    "api_name": "stock_basic",
                    "symbols": ["000001.SZ", "000002.SZ"],
                    "fields": ["ts_code", "symbol", "name", "list_date"],
                }
            ]
        },
        ToolContext(run_id="r-symbol-only-coverage"),
    )

    assert result["coverage_status"] == "NOT_VERIFIED"
    assert result["local_coverage_status"] == "PARTIAL_COVERAGE"
    assert result["local_coverage"][0]["missing_symbols"] == ["000002.SZ"]
    assert result["local_coverage"][0]["partial_reasons"] == ["missing_symbols"]


def test_plan_tushare_fetch_returns_quota_aware_payload_for_large_agent_plan(
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    plan_tool = _tools(tmp_path, lake)["plan_tushare_fetch"]

    result = plan_tool.run(
        {
            "items": [
                {
                    "api_name": "fina_indicator",
                    "symbols": [f"{index:06d}.SZ" for index in range(1, 50)],
                    "fields": ["ts_code", "end_date", "roe"],
                    "start_date": "20240101",
                    "end_date": "20241231",
                }
            ]
        },
        ToolContext(run_id="r-quota-plan", requested_by_llm=True),
    )

    assert result["status"] == "planned"
    assert result["estimated_request_count"] == 49
    assert result["quota_profile"]["points"] == 2000
    assert result["quota_profile"]["max_requests_per_minute"] == 200
    assert result["budget_decision"]["status"] == "APPROVED_BY_ACCOUNT_QUOTA"
    assert "25" not in str(result.get("message", ""))


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
    assert result["metadata"]["budget_decision"]["status"] == "APPROVED_BY_ACCOUNT_QUOTA"
    assert not lake.dataset_path("raw", "tushare/daily_basic").exists()


def test_plan_tushare_fetch_classifies_corrupt_local_ledger(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = lake.root / "metadata" / "tushare_usage_ledger.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"PAR1broken-ledger-pagePAR1")

    result = _tools(tmp_path, lake)["plan_tushare_fetch"].run(
        {
            "items": [
                {
                    "api_name": "fina_indicator",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "end_date", "roe"],
                    "start_date": "20240101",
                    "end_date": "20241231",
                }
            ]
        },
        ToolContext(run_id="r-corrupt-ledger"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "TUSHARE_USAGE_LEDGER_CORRUPT"
    assert result["source"] == "local_metadata"
    assert result["remote_request_attempted"] is False
    assert result["execution_status"] == "ERROR"
    assert result["domain_status"] == "FAILED"
    assert result["evidence_status"] == "INVALID"
    assert result["recommendation_status"] == "BLOCKED"
    assert result["blockers"] == ["tushare_usage_ledger_corrupt"]
    assert result["repair_action"]["command"].endswith("--quarantine-corrupt")
    assert "remote_query_failed" not in str(result)


def test_corrupt_ledger_reason_is_persisted_in_tool_audit(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = lake.root / "metadata" / "tushare_usage_ledger.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"PAR1broken-ledger-pagePAR1")
    dependencies = _deps(tmp_path, lake)
    registry = AgentToolRegistry(audit_logger=dependencies.audit_logger)
    registry.register_all(*build_remote_data_tools(dependencies))

    result = registry.run_tool(
        "plan_tushare_fetch",
        {
            "items": [
                {
                    "api_name": "fina_indicator",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "end_date", "roe"],
                    "start_date": "20240101",
                    "end_date": "20241231",
                }
            ]
        },
        ToolContext(run_id="r-corrupt-ledger-audit"),
    )

    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result["reason"] == "TUSHARE_USAGE_LEDGER_CORRUPT"
    assert audit["output_data"]["reason"] == "TUSHARE_USAGE_LEDGER_CORRUPT"
    assert "tushare_usage_ledger_corrupt" in audit["blockers"]


def test_run_tushare_fetch_does_not_contact_remote_when_ledger_is_corrupt(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = lake.root / "metadata" / "tushare_usage_ledger.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"PAR1broken-ledger-pagePAR1")
    client = RecordingGenericClient({})
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: client,
    )

    result = _tools(tmp_path, lake)["run_tushare_fetch"].run(
        {
            "items": [
                {
                    "api_name": "fina_indicator",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "end_date", "roe"],
                    "start_date": "20240101",
                    "end_date": "20241231",
                }
            ],
            "execute_plan": True,
        },
        ToolContext(run_id="r-corrupt-ledger-live"),
    )

    assert result["reason"] == "TUSHARE_USAGE_LEDGER_CORRUPT"
    assert result["remote_request_attempted"] is False
    assert client.calls == []


def test_plan_discloses_usage_history_reset_after_explicit_quarantine(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    initialize_persistence(lake, migrate_legacy_ledger=False)
    legacy = lake.root / "metadata" / "tushare_usage_ledger.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"PAR1broken-ledger-pagePAR1")
    ledger = TushareUsageLedger.from_data_lake(lake)
    repair_tushare_usage_ledger(ledger, quarantine_corrupt=True)

    result = _tools(tmp_path, lake)["plan_tushare_fetch"].run(
        {
            "items": [
                {
                    "api_name": "fina_indicator",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "end_date", "roe"],
                    "start_date": "20240101",
                    "end_date": "20241231",
                }
            ]
        },
        ToolContext(run_id="r-history-reset"),
    )

    assert result["status"] == "planned"
    assert result["warnings"] == ["TUSHARE_USAGE_HISTORY_RESET"]


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

    assert result["status"] == "PARTIAL_UPDATE"
    assert result["domain_status"] == "PARTIAL"
    assert result["evidence_status"] == "INCOMPLETE"
    assert result["coverage_status"] == "PARTIAL_COVERAGE"
    assert result["dataset_results"][0]["partial_reasons"] == [
        "starts_after_requested_start",
        "ends_before_requested_end",
    ]
    assert result["plan"]["budget_decision"]["status"] == "APPROVED_BY_ACCOUNT_QUOTA"
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
    assert not (lake.root / "metadata" / "tushare_usage_ledger.parquet").exists()
    usage = lake.query_parquet(
        "SELECT status, execution_mode FROM tushare_usage_events_v1 ORDER BY status"
    )
    assert set(usage["status"]) == {"PLANNED", "SUCCESS"}
    assert set(usage["execution_mode"]) == {"autonomous"}


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
