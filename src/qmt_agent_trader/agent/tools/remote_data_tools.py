"""Registry-driven Tushare data tools for the research agent."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import pandas as pd

from qmt_agent_trader.agent.data_tool_outcome import (
    data_tool_invalid_request,
    data_tool_no_data,
    data_tool_not_configured,
    data_tool_ok,
    data_tool_weak,
)
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.data.providers.base import FetchItem
from qmt_agent_trader.data.providers.tushare.client import TushareClient
from qmt_agent_trader.data.providers.tushare.fetcher import TushareFetcher
from qmt_agent_trader.data.providers.tushare.planner import TusharePlannerConfig
from qmt_agent_trader.data.providers.tushare.provider import TushareProvider
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.table_builder import ALLOWED_SILVER_TABLES, DataTableBuilder

_lake: DataLake | None = None
_settings: Settings | None = None
_client_factory: Callable[[], TushareClient] | None = None
_lake_var: ContextVar[DataLake | None] = ContextVar("remote_data_tool_lake", default=None)
_settings_var: ContextVar[Settings | None] = ContextVar(
    "remote_data_tool_settings",
    default=None,
)
_client_factory_var: ContextVar[Callable[[], TushareClient] | None] = ContextVar(
    "remote_data_tool_client_factory",
    default=None,
)
_AUTONOMOUS_TUSHARE_FETCH_MAX_REQUESTS = 25


def wire(
    *,
    data_lake: DataLake,
    settings: Settings | None = None,
    client_factory: Callable[[], TushareClient] | None = None,
) -> None:
    global _lake, _settings, _client_factory
    _lake = data_lake
    _settings = settings or get_settings()
    _client_factory = client_factory


def _get_lake() -> DataLake | None:
    return _lake_var.get() or _lake


def _get_settings() -> Settings:
    return _settings_var.get() or _settings or get_settings()


def _build_client(settings: Settings) -> TushareClient:
    client_factory = _client_factory_var.get() or _client_factory
    if client_factory is not None:
        return client_factory()
    token = settings.tushare_token.get_secret_value() if settings.tushare_token else None
    return TushareClient(
        token=token,
        timeout_seconds=settings.remote_data_http_timeout_seconds,
    )


def _with_deps(
    deps: AgentToolDependencies,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    lake_token = _lake_var.set(deps.data_lake)
    settings_token = _settings_var.set(deps.settings)
    client_factory_token = _client_factory_var.set(None)
    try:
        return fn(input_data, context)
    finally:
        _client_factory_var.reset(client_factory_token)
        _settings_var.reset(settings_token)
        _lake_var.reset(lake_token)


def _tushare_provider(lake: DataLake | None = None) -> TushareProvider:
    settings = _get_settings()
    config = TusharePlannerConfig(
        symbol_fanout_threshold=30,
        autonomous_request_budget=_AUTONOMOUS_TUSHARE_FETCH_MAX_REQUESTS,
        manual_request_budget=500,
        max_days_per_batch=settings.remote_data_max_days_per_call,
    )
    if lake is None:
        return TushareProvider(planner_config=config)
    fetcher = TushareFetcher(
        _build_client(settings),
        lake,
        min_interval_seconds=settings.remote_data_min_interval_seconds,
        retry_attempts=settings.remote_data_retry_attempts,
        retry_backoff_seconds=settings.remote_data_retry_backoff_seconds,
    )
    return TushareProvider(fetcher=fetcher, planner_config=config)


def _list_tushare_capabilities(
    input_data: dict[str, Any],
    _context: ToolContext,
) -> dict[str, Any]:
    capability = _tushare_provider().list_capabilities(
        category=_optional_str(input_data.get("category")),
        asset_type=_optional_str(input_data.get("asset_type")),
    )
    return data_tool_ok(
        status="OK",
        source=capability.source,
        endpoints=capability.endpoints,
        coverage_status="NOT_VERIFIED",
        message="Tushare capabilities listed; no data coverage was verified.",
    )


def _plan_tushare_fetch(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    try:
        items = _parse_fetch_items(input_data)
    except ValueError as exc:
        return data_tool_invalid_request(message=str(exc), reason="invalid_fetch_items")
    lake = _get_lake()
    if lake is not None:
        items = _attach_trade_dates_for_marketwide_fetches(items, lake)
    plan = _tushare_provider().plan_fetch(
        items,
        requested_by_llm=context.requested_by_llm,
        storage_mode=str(input_data.get("storage_mode", "persistent")),
    )
    payload = plan.as_dict()
    _attach_plan_summary(payload, lake=lake)
    return payload


def _run_tushare_fetch(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return data_tool_not_configured(status="NOT_AVAILABLE", message="data lake not wired")
    settings = _get_settings()
    execute_plan = bool(input_data.get("execute_plan", False))
    dry_run = bool(input_data.get("dry_run", False))
    if not dry_run and not execute_plan:
        return data_tool_invalid_request(
            message="run_tushare_fetch requires execute_plan=true for live execution",
            reason="execute_plan_required",
        )
    if not dry_run and settings.tushare_token is None and _client_factory is None:
        return data_tool_not_configured(
            message="TUSHARE_TOKEN is required for live Tushare fetch",
            reason="missing_tushare_token",
        )
    try:
        items = _parse_fetch_items(input_data)
    except ValueError as exc:
        return data_tool_invalid_request(message=str(exc), reason="invalid_fetch_items")

    items = _attach_trade_dates_for_marketwide_fetches(items, lake)
    provider = _tushare_provider(lake)
    plan = provider.plan_fetch(
        items,
        requested_by_llm=context.requested_by_llm,
        storage_mode=str(input_data.get("storage_mode", "persistent")),
    )
    if plan.status != "planned":
        payload = plan.as_dict()
        _attach_plan_summary(payload, lake=lake)
        return payload
    result = provider.run_fetch(plan, execute_plan=execute_plan, dry_run=dry_run).as_dict()
    result["plan"] = plan.as_dict()
    result["dry_run"] = dry_run
    result["execute_plan"] = execute_plan
    return result


def _build_data_table(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return data_tool_not_configured(status="NOT_AVAILABLE", message="data lake not wired")
    table = str(input_data.get("table", ""))
    snapshot = _optional_str(input_data.get("snapshot_as_of_date"))
    result = DataTableBuilder(lake).build(table, snapshot_as_of_date=snapshot)
    result_extra = {
        key: value for key, value in result.items() if key not in {"status", "message", "reason"}
    }
    if result.get("status") == "INVALID_REQUEST":
        return data_tool_invalid_request(
            message=str(result.get("message") or "invalid table"),
            reason="invalid_silver_table",
            **result_extra,
        )
    rows = int(result.get("rows", 0) or 0)
    if rows <= 0:
        return data_tool_no_data(
            status=str(result.get("status", "built")),
            message=f"{table} built with 0 rows.",
            reason="zero_rows_built",
            **result_extra,
        )
    return data_tool_weak(
        status=str(result.get("status", "built")),
        message=f"{table} built with {rows} rows; query tools must verify research coverage.",
        **result_extra,
    )


def _attach_plan_summary(payload: dict[str, Any], *, lake: DataLake | None) -> None:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return
    payload["strategy"] = items[0].get("strategy")
    payload["batches"] = [batch for item in items for batch in item.get("batches", [])]
    payload["target_tables"] = [
        target for item in items for target in item.get("target_tables", [])
    ]
    payload["wide_table_targets"] = sorted(
        {
            target
            for item in items
            for target in item.get("wide_table_targets", [])
        }
    )
    if lake is None:
        payload["local_coverage_status"] = "UNKNOWN"
        return
    local_coverage = [_local_coverage_for_planned_item(item, lake) for item in items]
    payload["local_coverage"] = local_coverage
    payload["local_coverage_status"] = _aggregate_local_coverage(local_coverage)


def _local_coverage_for_planned_item(
    item: dict[str, Any],
    lake: DataLake,
) -> dict[str, Any]:
    target_dataset = str(item.get("target_dataset", ""))
    dataset_id = str(item.get("dataset_id", ""))
    path = lake.dataset_path("raw", target_dataset)
    params = dict(item.get("params", {}))
    start, end = _coverage_bounds_from_params(params)
    coverage: dict[str, Any] = {
        "dataset_id": dataset_id,
        "target_dataset": target_dataset,
        "coverage_start": start,
        "coverage_end": end,
        "next_repair_tool": "run_tushare_fetch",
    }
    if not path.exists():
        return {**coverage, "status": "NO_DATA", "reason": "raw_dataset_missing"}
    frame = lake.read_parquet("raw", target_dataset)
    if frame.empty:
        return {**coverage, "status": "NO_DATA", "reason": "raw_dataset_empty"}

    symbols = [str(item) for item in item.get("symbols", [])]
    missing_symbols: list[str] = []
    scoped = frame
    if symbols and "ts_code" in scoped.columns:
        present_symbols = set(scoped["ts_code"].astype(str))
        missing_symbols = [symbol for symbol in symbols if symbol not in present_symbols]
        scoped = scoped[scoped["ts_code"].astype(str).isin(symbols)]
        if scoped.empty:
            return {
                **coverage,
                "status": "NO_DATA",
                "reason": "requested_symbols_missing",
                "missing_symbols": missing_symbols,
            }

    date_column = _coverage_date_column(item, scoped)
    if date_column is None or start is None or end is None:
        symbol_partial_reasons = ["missing_symbols"] if missing_symbols else []
        return {
            **coverage,
            "status": (
                "PARTIAL_COVERAGE" if symbol_partial_reasons else "LOCAL_DATA_PRESENT"
            ),
            "rows": len(scoped),
            "missing_symbols": missing_symbols,
            "date_column": date_column,
            "partial_reasons": symbol_partial_reasons,
            "reason": "coverage_range_not_derivable" if date_column is None else None,
        }

    comparable = scoped[date_column].astype(str).str.replace("-", "", regex=False)
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")
    in_range = scoped[(comparable >= start_key) & (comparable <= end_key)]
    if in_range.empty:
        return {
            **coverage,
            "status": "NO_DATA",
            "reason": "no_rows_in_requested_range",
            "missing_symbols": missing_symbols,
            "date_column": date_column,
        }

    actual_values = in_range[date_column].astype(str).str.replace("-", "", regex=False)
    actual_start = str(actual_values.min())
    actual_end = str(actual_values.max())
    partial_reasons: list[str] = []
    if actual_start > start_key:
        partial_reasons.append("starts_after_requested_start")
    if actual_end < end_key:
        partial_reasons.append("ends_before_requested_end")
    if missing_symbols:
        partial_reasons.append("missing_symbols")
    status = "PARTIAL_COVERAGE" if partial_reasons else "LOCAL_DATA_PRESENT"
    return {
        **coverage,
        "status": status,
        "rows": len(in_range),
        "actual_start": actual_start,
        "actual_end": actual_end,
        "missing_symbols": missing_symbols,
        "date_column": date_column,
        "partial_reasons": partial_reasons,
    }


def _aggregate_local_coverage(items: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status")) for item in items}
    if "NO_DATA" in statuses:
        return "NO_DATA"
    if "PARTIAL_COVERAGE" in statuses:
        return "PARTIAL_COVERAGE"
    if statuses == {"LOCAL_DATA_PRESENT"}:
        return "LOCAL_DATA_PRESENT"
    return "UNKNOWN"


def _coverage_date_column(item: dict[str, Any], frame: pd.DataFrame) -> str | None:
    candidates = [
        "trade_date",
        "date",
        "cal_date",
        "month",
        "quarter",
        "end_date",
        "ann_date",
    ]
    declared = set(item.get("key_columns", [])) | set(item.get("fields", []))
    for column in candidates:
        if column in declared and column in frame.columns:
            return column
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _coverage_bounds_from_params(params: dict[str, Any]) -> tuple[str | None, str | None]:
    for start_key, end_key in (
        ("start_date", "end_date"),
        ("start_m", "end_m"),
        ("start_q", "end_q"),
    ):
        start = _optional_str(params.get(start_key))
        end = _optional_str(params.get(end_key))
        if start or end:
            return start, end
    for point_key in ("trade_date", "date", "cal_date", "m", "q", "period", "ann_date"):
        value = _optional_str(params.get(point_key))
        if value:
            return value, value
    return None, None


def _attach_trade_dates_for_marketwide_fetches(
    items: list[FetchItem],
    lake: DataLake,
) -> list[FetchItem]:
    provider = _tushare_provider()
    config = TusharePlannerConfig(
        autonomous_request_budget=_AUTONOMOUS_TUSHARE_FETCH_MAX_REQUESTS
    )
    enriched: list[FetchItem] = []
    for item in items:
        spec = provider.registry.get(item.api_name)
        if (
            spec is None
            or not spec.supports_marketwide_by_date
            or len(item.symbols) <= config.symbol_fanout_threshold
            or not item.start_date
            or not item.end_date
            or item.trade_date
            or item.params.get("trade_dates")
        ):
            enriched.append(item)
            continue
        trade_dates = _new_layout_trade_dates(lake, item.start_date, item.end_date)
        if not trade_dates:
            enriched.append(item)
            continue
        enriched.append(
            FetchItem(
                api_name=item.api_name,
                symbols=item.symbols,
                fields=item.fields,
                start_date=item.start_date,
                end_date=item.end_date,
                trade_date=item.trade_date,
                params={**item.params, "trade_dates": trade_dates},
            )
        )
    return enriched


def _new_layout_trade_dates(lake: DataLake, start: str, end: str) -> list[str]:
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")
    frame = None
    if lake.dataset_path("silver", "trade_calendar").exists():
        frame = lake.read_parquet("silver", "trade_calendar")
    elif lake.dataset_path("raw", "tushare/trade_cal").exists():
        frame = lake.read_parquet("raw", "tushare/trade_cal")
    if frame is None or frame.empty or "cal_date" not in frame.columns:
        return []
    data = frame.copy()
    if "is_open" in data.columns:
        data = data[data["is_open"].astype(str).isin({"1", "True", "true"})]
    dates = data["cal_date"].astype(str).str.replace("-", "", regex=False)
    return sorted(date for date in dates if start_key <= date <= end_key)


def _parse_fetch_items(input_data: dict[str, Any]) -> list[FetchItem]:
    raw_items = input_data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("items must be a non-empty list")
    items: list[FetchItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("each fetch item must be an object")
        api_name = raw.get("api_name")
        if not isinstance(api_name, str) or not api_name:
            raise ValueError("each fetch item requires api_name")
        raw_params = raw.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        items.append(
            FetchItem(
                api_name=api_name,
                symbols=_normalize_symbols(raw.get("symbols")),
                fields=_normalize_fields(raw.get("fields")),
                trade_date=_optional_str(raw.get("trade_date")),
                start_date=_optional_str(raw.get("start_date")),
                end_date=_optional_str(raw.get("end_date")),
                params=params,
            )
        )
    return items


def _normalize_symbols(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if "." not in text and text.isdigit() and len(text) == 6:
            text = f"{text}.SZ" if text.startswith(("0", "1", "2", "3")) else f"{text}.SH"
        if text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_fields(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("fields must be a list or comma-separated string")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_FETCH_ITEMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "api_name": {"type": "string"},
            "symbols": {"type": "array", "items": {"type": "string"}},
            "fields": {"type": "array", "items": {"type": "string"}},
            "trade_date": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "params": {"type": "object", "additionalProperties": True},
        },
        "required": ["api_name"],
        "additionalProperties": False,
    },
}


list_tushare_capabilities_tool: AgentTool = tool(
    ToolSpec(
        name="list_tushare_capabilities",
        description="列出 registry 中可用的 Tushare endpoint、字段、参数、主键、分页和 PIT 规则。",
        permission=PermissionLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "asset_type": {"type": "string"},
            },
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        deterministic=True,
    ),
    fn=_list_tushare_capabilities,
)


plan_tushare_fetch_tool: AgentTool = tool(
    ToolSpec(
        name="plan_tushare_fetch",
        description="校验结构化 Tushare fetch spec，并生成 planner-controlled 抓取计划。",
        permission=PermissionLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "items": _FETCH_ITEMS_SCHEMA,
                "storage_mode": {"type": "string"},
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        deterministic=True,
    ),
    fn=_plan_tushare_fetch,
)


run_tushare_fetch_tool: AgentTool = tool(
    ToolSpec(
        name="run_tushare_fetch",
        description=(
            "执行已规划的 Tushare fetch。dry_run=true 不访问远端；真实执行必须显式 "
            "execute_plan=true，并写入 registry 指定的新 raw layout。"
        ),
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_formal",
        input_schema={
            "type": "object",
            "properties": {
                "items": _FETCH_ITEMS_SCHEMA,
                "storage_mode": {"type": "string"},
                "dry_run": {"type": "boolean"},
                "execute_plan": {"type": "boolean"},
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        deterministic=False,
        timeout_seconds=300,
    ),
    fn=_run_tushare_fetch,
)


build_data_table_tool: AgentTool = tool(
    ToolSpec(
        name="build_data_table",
        description="从新 raw layout 构建允许的 silver 表，不创建 research_daily_wide。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_formal",
        input_schema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": sorted(ALLOWED_SILVER_TABLES)},
                "snapshot_as_of_date": {"type": "string"},
            },
            "required": ["table"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        deterministic=False,
    ),
    fn=_build_data_table,
)


def build_remote_data_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    return [
        tool(
            list_tushare_capabilities_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _list_tushare_capabilities, input_data, context
            ),
        ),
        tool(
            plan_tushare_fetch_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _plan_tushare_fetch, input_data, context
            ),
        ),
        tool(
            run_tushare_fetch_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _run_tushare_fetch, input_data, context
            ),
        ),
        tool(
            build_data_table_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _build_data_table, input_data, context
            ),
        ),
    ]
