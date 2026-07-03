from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.remote_data_tools import (
    build_remote_data_tools,
    plan_tushare_fetch_tool,
    run_tushare_fetch_tool,
    wire,
)
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.providers.base import FetchItem
from qmt_agent_trader.data.providers.tushare.client import TushareClient
from qmt_agent_trader.data.providers.tushare.fetcher import TushareFetcher
from qmt_agent_trader.data.providers.tushare.planner import TushareFetchPlanner
from qmt_agent_trader.data.providers.tushare.registry import default_tushare_registry
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.table_builder import DataTableBuilder


class FakeGenericClient(TushareClient):
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        super().__init__(token="fake")
        self.frames = frames
        self.calls: list[tuple[str, dict[str, object], list[str] | None]] = []

    def query(
        self,
        api_name: str,
        params: dict[str, object],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        self.calls.append((api_name, params, fields))
        return self.frames.get(api_name, pd.DataFrame()).copy()


def test_tushare_endpoint_registry_loads_official_inventory() -> None:
    registry = default_tushare_registry()
    daily = registry.require("daily")
    daily_basic = registry.require("daily_basic")

    assert daily.description is not None
    assert daily.field_descriptions["ts_code"] == "股票代码"
    assert daily.field_descriptions["trade_date"] == "交易日期"
    assert daily.call_limit["status"] == "DOCUMENTED"
    assert daily.call_limit["requests_per_minute"] == 500
    assert daily.call_limit["rows_per_request"] == 6000
    assert daily_basic.implemented is True
    assert "pe_ttm" in daily_basic.fields
    assert daily_basic.field_descriptions["pe_ttm"].startswith("市盈率")
    assert daily_basic.call_limit["status"] == "DOCUMENTED"
    assert daily_basic.call_limit["rows_per_request"] == 6000
    assert daily_basic.key_columns == ("ts_code", "trade_date")
    assert daily_basic.raw_dataset_name == "tushare/daily_basic"
    assert registry.require("repurchase").implemented is False


def test_tushare_capabilities_expose_metadata_completeness_gaps() -> None:
    registry = default_tushare_registry()
    capabilities = {item["api_name"]: item for item in registry.as_capabilities()}

    assert capabilities["daily"]["field_description_status"] == "COMPLETE"
    assert capabilities["daily"]["missing_field_descriptions"] == []
    assert capabilities["daily"]["call_limit"]["status"] == "DOCUMENTED"
    assert capabilities["suspend_d"]["field_description_status"] == "UNKNOWN"
    assert capabilities["suspend_d"]["missing_field_descriptions"] == [
        "ts_code",
        "trade_date",
        "suspend_timing",
        "suspend_type",
    ]
    assert capabilities["suspend_d"]["call_limit"]["status"] == "UNKNOWN"


def test_tushare_planner_rejects_unknown_placeholder_field_and_symbol() -> None:
    planner = TushareFetchPlanner()

    placeholder = planner.plan([FetchItem(api_name="repurchase")])
    unknown_field = planner.plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=["000001.SZ"],
                fields=["ts_code", "市盈率"],
                start_date="20240101",
                end_date="20240131",
            )
        ]
    )
    invalid_symbol = planner.plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=["平安银行"],
                fields=["ts_code", "trade_date"],
                start_date="20240101",
                end_date="20240131",
            )
        ]
    )

    assert placeholder.status == "NOT_IMPLEMENTED"
    assert placeholder.reason == "endpoint_registered_as_placeholder"
    assert unknown_field.status == "INVALID_REQUEST"
    assert unknown_field.reason == "unknown_fields"
    assert invalid_symbol.status == "INVALID_REQUEST"
    assert invalid_symbol.reason == "invalid_ts_code"


def test_tushare_planner_supports_multi_symbol_fanout_and_budget_block() -> None:
    planner = TushareFetchPlanner()

    plan = planner.plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=["000001.SZ", "600519.SH"],
                fields=["ts_code", "trade_date", "pe_ttm", "pb"],
                start_date="20240101",
                end_date="20240131",
            )
        ],
        requested_by_llm=True,
    )
    blocked = planner.plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=[f"{index:06d}.SZ" for index in range(1, 40)],
                fields=["ts_code", "trade_date"],
                start_date="20240101",
                end_date="20240131",
            )
        ],
        requested_by_llm=True,
    )

    assert plan.status == "planned"
    assert plan.estimated_request_count == 2
    assert plan.items[0]["strategy"] == "fanout_by_symbol_range"
    assert blocked.status == "BLOCKED"
    assert blocked.reason == "TRADE_CALENDAR_REQUIRED"


def test_tushare_planner_uses_explicit_trade_dates_for_large_marketwide_fetch() -> None:
    planner = TushareFetchPlanner()

    plan = planner.plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=[f"{index:06d}.SZ" for index in range(1, 40)],
                fields=["ts_code", "trade_date", "pe_ttm"],
                start_date="20240101",
                end_date="20240105",
                params={"trade_dates": ["20240102", "20240103", "20240105"]},
            )
        ],
        requested_by_llm=True,
    )

    assert plan.status == "planned"
    assert plan.items[0]["strategy"] == "marketwide_by_trade_date"
    assert [batch["params"]["trade_date"] for batch in plan.items[0]["batches"]] == [
        "20240102",
        "20240103",
        "20240105",
    ]


def test_tushare_fetcher_writes_new_raw_layout_and_metadata(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "pe_ttm": 5.0,
                "pb": 0.8,
            }
        ]
    )
    client = FakeGenericClient({"daily_basic": frame})
    planner = TushareFetchPlanner()
    plan = planner.plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=["000001.SZ"],
                fields=["ts_code", "trade_date", "pe_ttm", "pb"],
                start_date="20240101",
                end_date="20240131",
            )
        ]
    )

    result = TushareFetcher(client, lake, sleep=lambda _: None).run(
        plan,
        execute_plan=True,
    )

    assert result.status == "updated"
    assert lake.dataset_path("raw", "tushare/daily_basic").exists()
    assert not lake.dataset_path("raw", "tushare_daily_basic").exists()
    assert client.calls[0][0] == "daily_basic"
    state = lake.query_parquet("SELECT * FROM data_fetch_state_v2").to_dict(orient="records")
    assert state[0]["dataset_id"] == "tushare.daily_basic"


def test_tushare_fetcher_records_point_parameter_coverage(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "pe_ttm": 5.0,
            }
        ]
    )
    client = FakeGenericClient({"daily_basic": frame})
    plan = TushareFetchPlanner().plan(
        [
            FetchItem(
                api_name="daily_basic",
                fields=["ts_code", "trade_date", "pe_ttm"],
                trade_date="20240102",
            )
        ]
    )

    result = TushareFetcher(client, lake, sleep=lambda _: None).run(
        plan,
        execute_plan=True,
    )

    assert result.status == "updated"
    state = lake.query_parquet("SELECT * FROM data_fetch_state_v2").to_dict(orient="records")
    assert state[0]["coverage_start"] == "20240102"
    assert state[0]["coverage_end"] == "20240102"


def test_tushare_fetcher_rejects_schema_mismatch_without_write(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeGenericClient({"daily_basic": pd.DataFrame([{"ts_code": "000001.SZ"}])})
    plan = TushareFetchPlanner().plan(
        [
            FetchItem(
                api_name="daily_basic",
                symbols=["000001.SZ"],
                fields=["ts_code", "trade_date"],
                start_date="20240101",
                end_date="20240131",
            )
        ]
    )

    result = TushareFetcher(client, lake, sleep=lambda _: None).run(plan, execute_plan=True)

    assert result.status == "error"
    assert result.errors[0]["status"] == "SCHEMA_MISMATCH"
    assert not lake.dataset_path("raw", "tushare/daily_basic").exists()


def test_build_data_table_keeps_macro_long_and_financial_pit(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_incremental_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240401",
                    "f_ann_date": "20240402",
                    "end_date": "20231231",
                    "report_type": "1",
                    "total_revenue": 10.0,
                }
            ]
        ),
        "raw",
        "tushare/income",
        key_columns=["ts_code", "end_date", "ann_date", "report_type"],
    )
    lake.write_incremental_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240401",
                    "f_ann_date": "20240403",
                    "end_date": "20231231",
                    "total_assets": 100.0,
                }
            ]
        ),
        "raw",
        "tushare/balancesheet",
        key_columns=["ts_code", "end_date", "ann_date"],
    )
    lake.write_incremental_parquet(
        pd.DataFrame([{"month": "202401", "nt_val": 102.0, "nt_yoy": 1.2}]),
        "raw",
        "tushare/cn_cpi",
        key_columns=["month"],
    )

    reports = DataTableBuilder(lake).build("financial_reports_wide")
    macro = DataTableBuilder(lake).build("macro_series")

    assert reports["status"] == "built"
    assert macro["status"] == "built"
    financial = lake.read_parquet("silver", "financial_reports_wide")
    assert len(financial) == 1
    assert financial.iloc[0]["visible_date"] == "20240402"
    assert financial.iloc[0]["total_revenue"] == 10.0
    assert financial.iloc[0]["total_assets"] == 100.0
    assert financial.iloc[0]["source_flags"] == "tushare.balancesheet,tushare.income"
    macro_frame = lake.read_parquet("silver", "macro_series")
    assert set(macro_frame["macro_id"]) == {"cn_cpi.nt_val", "cn_cpi.nt_yoy"}
    assert "ts_code" not in macro_frame.columns


def test_agent_visible_remote_data_tools_use_new_provider_surface(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    deps = AgentToolDependencies(
        settings=Settings(project_root=tmp_path, tushare_token=None),
        data_lake=lake,
        sandbox=CodeSandbox(tmp_path / "generated"),
        experiment_store=ExperimentStore(tmp_path / "experiments"),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )

    names = {item.spec.name for item in build_remote_data_tools(deps)}

    assert names == {
        "list_tushare_capabilities",
        "plan_tushare_fetch",
        "run_tushare_fetch",
        "build_data_table",
    }


def test_run_tushare_fetch_tool_dry_run_never_contacts_client(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeGenericClient({})
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path),
        client_factory=lambda: client,
    )

    result = run_tushare_fetch_tool.run(
        {
            "items": [
                {
                    "api_name": "daily_basic",
                    "symbols": ["000001.SZ"],
                    "fields": ["ts_code", "trade_date"],
                    "start_date": "20240101",
                    "end_date": "20240131",
                }
            ],
            "dry_run": True,
        },
        ToolContext(run_id="dry-run", requested_by_llm=True),
    )

    assert result["status"] == "planned"
    assert client.calls == []


def test_plan_tushare_fetch_tool_injects_new_layout_trade_calendar_only(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(data_lake=lake, settings=Settings(project_root=tmp_path))
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

    blocked = plan_tushare_fetch_tool.run(
        request,
        ToolContext(run_id="plan-without-new-calendar", requested_by_llm=True),
    )
    lake.write_parquet(
        pd.DataFrame([{"cal_date": "20240102", "is_open": 1}]),
        "raw",
        "tushare_trade_calendar",
    )
    still_blocked = plan_tushare_fetch_tool.run(
        request,
        ToolContext(run_id="plan-with-legacy-calendar", requested_by_llm=True),
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"cal_date": "20240102", "is_open": 1},
                {"cal_date": "20240103", "is_open": 1},
                {"cal_date": "20240104", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    planned = plan_tushare_fetch_tool.run(
        request,
        ToolContext(run_id="plan-with-new-calendar", requested_by_llm=True),
    )

    assert blocked["status"] == "BLOCKED"
    assert blocked["reason"] == "TRADE_CALENDAR_REQUIRED"
    assert still_blocked["status"] == "BLOCKED"
    assert planned["status"] == "planned"
    assert planned["strategy"] == "marketwide_by_trade_date"
    assert [batch["params"]["trade_date"] for batch in planned["batches"]] == [
        "20240102",
        "20240103",
    ]
