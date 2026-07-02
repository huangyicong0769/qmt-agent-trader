"""Controlled remote data update tools for the research agent."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import date, datetime, timedelta
from typing import Any

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.data.macro import MACRO_DATASETS
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient
from qmt_agent_trader.services.data_update_service import (
    FINANCIAL_TABLES,
    RequestLimiter,
    TushareDataUpdateService,
    build_data_update_plan,
    build_fundamental_update_plan,
    build_macro_update_plan,
)

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
_AUTONOMOUS_REMOTE_UPDATE_MAX_REQUESTS = 25


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


def _timeout_with_deps(
    deps: AgentToolDependencies,
    input_data: dict[str, Any],
    context: ToolContext,
) -> int:
    lake_token = _lake_var.set(deps.data_lake)
    settings_token = _settings_var.set(deps.settings)
    try:
        return _remote_data_update_timeout_seconds(input_data, context)
    finally:
        _settings_var.reset(settings_token)
        _lake_var.reset(lake_token)


def _plan_remote_data_update(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
    try:
        source, start, end = _parse_request(input_data)
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        ts_code = _normalize_ts_code(input_data.get("ts_code"))
        symbols = _normalize_symbols(input_data.get("symbols"))
        asset_type = str(input_data.get("asset_type", "stock")).lower()
        effective_start, metadata = _effective_start_from_local_basics(
            lake, start, ts_code=ts_code, asset_type=asset_type
        )
        if symbols:
            metadata["requested_symbols_count"] = len(symbols)
        if _parse_date(effective_start) > _parse_date(end):
            metadata["reason"] = "requested_end_before_listing"
            metadata["plan_meaning"] = "no_data_expected_before_security_listing"
            return {
                "status": "NO_DATA_EXPECTED",
                "source": source,
                "start_date": effective_start,
                "end_date": end,
                "data_update_needed": False,
                "metadata": metadata,
                "missing_ranges": [],
                "requests": [],
            }
        expected_dates, calendar_source = _expected_coverage_dates(
            lake,
            effective_start,
            end,
        )
        uses_date_calendar = calendar_source in {
            "tushare_trade_calendar",
            "observed_market_daily_dates",
        }
        missing_ranges = _missing_ranges(
            lake,
            expected_dates,
            ts_code=ts_code,
            symbols=symbols,
            asset_type=asset_type,
        )
        coverage = _symbol_coverage(
            lake,
            expected_dates,
            ts_code=ts_code,
            symbols=symbols,
            asset_type=asset_type,
        )
        missing_ranges = coverage.get("missing_ranges", missing_ranges)
        estimated_request_count = _estimate_request_count(
            input_data,
            missing_dates_count=int(coverage.get("missing_dates_count", 0)),
            data_update_needed=bool(missing_ranges),
            scoped=bool(ts_code),
        )
        metadata["plan_meaning"] = "dry_run_only_no_remote_fetch_performed"
        metadata["calendar_source"] = calendar_source
        metadata["missing_ranges_are_calendar_days"] = not uses_date_calendar
        metadata["requires_trade_calendar_validation"] = not uses_date_calendar
        if not uses_date_calendar:
            metadata["warning"] = (
                "missing_ranges are calendar-day gaps; do not claim they are "
                "weekends or holidays without trade-calendar validation"
            )
        actual_coverage = _actual_data_coverage(lake, ts_code=ts_code, symbols=symbols)
        data_freshness = (
            "covers_expected_trading_dates"
            if not missing_ranges
            else "missing_expected_trading_dates"
        )
        coverage_end = actual_coverage.get("actual_data_end")
        status = (
            "CALENDAR_VALIDATION_REQUIRED"
            if missing_ranges and not uses_date_calendar
            else "planned"
        )
        metadata.update(actual_coverage)
        metadata.update(
            {
                "coverage_by_symbol": coverage["coverage_by_symbol"],
                "missing_symbols": coverage["missing_symbols"],
                "stale_symbols": coverage["stale_symbols"],
                "covered_symbols": coverage["covered_symbols"],
                "estimated_request_count": estimated_request_count,
            }
        )
        metadata["data_freshness"] = data_freshness
        return {
            "status": status,
            "source": source,
            "start_date": effective_start,
            "end_date": end,
            "requested_start_date": start,
            "requested_end_date": end,
            "actual_data_start": actual_coverage.get("actual_data_start"),
            "actual_data_end": coverage_end,
            "coverage_start_date": actual_coverage.get("actual_data_start"),
            "coverage_end_date": coverage_end,
            "data_freshness": data_freshness,
            "data_update_needed": bool(missing_ranges),
            "metadata": metadata,
            "missing_ranges": missing_ranges,
            "coverage_by_symbol": coverage["coverage_by_symbol"],
            "missing_symbols": coverage["missing_symbols"],
            "stale_symbols": coverage["stale_symbols"],
            "covered_symbols": coverage["covered_symbols"],
            "estimated_request_count": estimated_request_count,
            "requests": build_data_update_plan(TushareClient(token=None), effective_start, end),
        }
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}


def _run_remote_data_update(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}

    settings = _get_settings()
    timeout_seconds_used = _remote_data_update_timeout_seconds(input_data, context)
    try:
        source, start, end = _parse_request(input_data)
        ts_code = _normalize_ts_code(input_data.get("ts_code"))
        symbols = _normalize_symbols(input_data.get("symbols"))
        asset_type = str(input_data.get("asset_type", "stock")).lower()
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        if asset_type not in {"stock", "etf", "auto"}:
            return {
                "status": "INVALID_REQUEST",
                "message": "asset_type must be stock, etf, or auto",
            }
        if _needs_auto_chunk(start, end, settings) and bool(input_data.get("auto_chunk", False)):
            return _run_chunked_remote_data_update(
                input_data,
                context,
                settings=settings,
                start=start,
                end=end,
                ts_code=ts_code,
                symbols=symbols,
                asset_type=asset_type,
                timeout_seconds_used=timeout_seconds_used,
            )
        if ts_code is None:
            _validate_span(start, end, settings.remote_data_max_days_per_call)
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    if bool(input_data.get("dry_run", False)):
        planned = _plan_remote_data_update(input_data, context)
        planned["dry_run"] = True
        planned["timeout_seconds_used"] = timeout_seconds_used
        return planned

    planned = _plan_remote_data_update({**input_data, "dry_run": True}, context)
    if planned.get("data_update_needed") is False:
        planned_metadata = planned.get("metadata")
        metadata = planned_metadata if isinstance(planned_metadata, dict) else {}
        return {
            **planned,
            "status": "up_to_date",
            "dry_run": False,
            "timeout_seconds_used": timeout_seconds_used,
            "metadata": {
                **metadata,
                "live_fetch_skipped": True,
                "skip_reason": "requested_range_already_covered",
            },
        }
    if (
        context.requested_by_llm
        and _estimated_request_count(planned) > _AUTONOMOUS_REMOTE_UPDATE_MAX_REQUESTS
    ):
        return {
            "status": "BLOCKED",
            "reason": "AUTONOMOUS_REMOTE_UPDATE_TOO_LARGE",
            "message": (
                "Autonomous agent remote data update would require too many requests; "
                "use dry_run planning, narrower symbols/date windows, or dedicated "
                "fundamental/macro chunked update tools before live execution."
            ),
            "estimated_request_count": _estimated_request_count(planned),
            "max_autonomous_request_count": _AUTONOMOUS_REMOTE_UPDATE_MAX_REQUESTS,
            "timeout_seconds_used": timeout_seconds_used,
            "missing_ranges": planned.get("missing_ranges", []),
            "missing_symbols": planned.get("missing_symbols", []),
            "next_repair_tool": _next_remote_repair_tool(input_data),
        }

    if settings.tushare_token is None and _client_factory is None:
        return {
            "status": "NOT_CONFIGURED",
            "message": "TUSHARE_TOKEN is required for live remote data update",
        }

    try:
        client = _build_client(settings)
        service = TushareDataUpdateService(
            client,
            lake,
            limiter=RequestLimiter(
                min_interval_seconds=settings.remote_data_min_interval_seconds
            ),
            lock_timeout_seconds=settings.remote_data_lock_timeout_seconds,
            retry_attempts=settings.remote_data_retry_attempts,
            retry_backoff_seconds=settings.remote_data_retry_backoff_seconds,
        )
        result = service.update(
            start,
            end,
            include_daily=bool(input_data.get("include_daily", True)),
            include_basics=bool(input_data.get("include_basics", True)),
            ts_code=ts_code,
            asset_type=asset_type,
            required_symbols=symbols or None,
        )
        payload = result.as_dict()
        actual_coverage = _actual_data_coverage(lake, ts_code=ts_code, symbols=symbols)
        payload.update(
            {
                "requested_start_date": start,
                "requested_end_date": end,
                "actual_data_start": actual_coverage.get("actual_data_start"),
                "actual_data_end": actual_coverage.get("actual_data_end"),
                "coverage_start_date": actual_coverage.get("actual_data_start"),
                "coverage_end_date": actual_coverage.get("actual_data_end"),
                "timeout_seconds_used": timeout_seconds_used,
            }
        )
        payload_metadata = payload.get("metadata")
        if isinstance(payload_metadata, dict):
            payload_metadata.update(actual_coverage)
        else:
            payload["metadata"] = dict(actual_coverage)
            payload_metadata = payload["metadata"]
        post_update = _plan_remote_data_update({**input_data, "dry_run": True}, context)
        _copy_coverage_fields(payload, post_update)
        if isinstance(payload_metadata, dict):
            payload_metadata.update(
                {
                    "coverage_by_symbol": post_update.get("coverage_by_symbol", {}),
                    "missing_symbols": post_update.get("missing_symbols", []),
                    "stale_symbols": post_update.get("stale_symbols", []),
                    "covered_symbols": post_update.get("covered_symbols", []),
                    "estimated_request_count": post_update.get("estimated_request_count", 0),
                }
            )
        if post_update.get("data_update_needed") is True:
            payload["status"] = "PARTIAL_COVERAGE"
            payload["data_update_needed"] = True
            if isinstance(payload_metadata, dict):
                payload_metadata["post_update_status"] = "PARTIAL_COVERAGE"
        else:
            payload["data_update_needed"] = False
        return payload
    except Exception as exc:
        message = _sanitize_error(str(exc), settings)
        lake.record_fetch_result(
            source=source,
            dataset="remote_data_update",
            start_date=start,
            end_date=end,
            status="error",
            row_count=0,
            checksum=None,
            error=message,
        )
        return {"status": "error", "message": message}


def _run_chunked_remote_data_update(
    input_data: dict[str, Any],
    context: ToolContext,
    *,
    settings: Settings,
    start: str,
    end: str,
    ts_code: str | None,
    symbols: list[str],
    asset_type: str,
    timeout_seconds_used: int,
) -> dict[str, Any]:
    chunks = _chunk_ranges(start, end, settings.remote_data_max_days_per_call)
    batch_plans = [
        _plan_remote_data_update(
            {
                **input_data,
                "start_date": chunk_start,
                "end_date": chunk_end,
                "auto_chunk": False,
                "dry_run": True,
            },
            context,
        )
        for chunk_start, chunk_end in chunks
    ]
    batches = [
        {
            **plan,
            "batch_index": index,
            "start_date": chunk_start,
            "end_date": chunk_end,
        }
        for index, (plan, (chunk_start, chunk_end)) in enumerate(
            zip(batch_plans, chunks, strict=True),
            start=1,
        )
    ]
    estimated_request_count = sum(_estimated_request_count(plan) for plan in batch_plans)
    execute_plan = bool(input_data.get("execute_plan", False))
    dry_run = bool(input_data.get("dry_run", False))
    missing_ranges = [
        item
        for batch in batches
        for item in batch.get("missing_ranges", [])
        if isinstance(item, dict)
    ]
    base_payload: dict[str, Any] = {
        "status": "planned",
        "category": "daily",
        "source": "tushare",
        "start_date": start,
        "end_date": end,
        "requested_start_date": start,
        "requested_end_date": end,
        "auto_chunk": True,
        "execute_plan": execute_plan,
        "dry_run": dry_run,
        "batches": batches,
        "coverage_status": _worst_coverage_status(
            [str(batch.get("coverage_status", "NO_DATA")) for batch in batches]
        ),
        "missing_ranges": missing_ranges,
        "remaining_missing_ranges": list(missing_ranges),
        "estimated_request_count": estimated_request_count,
        "timeout_seconds_used": timeout_seconds_used,
        "next_repair_tool": "run_remote_data_update",
    }
    if dry_run or not execute_plan:
        return base_payload

    if (
        context.requested_by_llm
        and len(chunks) > _AUTONOMOUS_REMOTE_UPDATE_MAX_REQUESTS
    ):
        return {
            **base_payload,
            "status": "BLOCKED",
            "reason": "AUTONOMOUS_REMOTE_UPDATE_TOO_LARGE",
            "message": (
                "Autonomous agent remote data update would require too many requests; "
                "use dry_run planning, narrower symbols/date windows, or dedicated "
                "fundamental/macro chunked update tools before live execution."
            ),
            "autonomous_batch_count": len(chunks),
            "max_autonomous_request_count": _AUTONOMOUS_REMOTE_UPDATE_MAX_REQUESTS,
        }

    if settings.tushare_token is None and _client_factory is None:
        return {
            **base_payload,
            "status": "NOT_CONFIGURED",
            "message": "TUSHARE_TOKEN is required for live remote data update",
        }

    batch_results: list[dict[str, Any]] = []
    try:
        service = _build_update_service(settings, _get_lake_required())
        for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            result = service.update(
                chunk_start,
                chunk_end,
                include_daily=bool(input_data.get("include_daily", True)),
                include_basics=bool(input_data.get("include_basics", True)),
                ts_code=ts_code,
                asset_type=asset_type,
                required_symbols=symbols or None,
            ).as_dict()
            result.update(
                {
                    "batch_index": index,
                    "start_date": chunk_start,
                    "end_date": chunk_end,
                }
            )
            batch_results.append(result)
    except Exception as exc:
        return {
            **base_payload,
            "status": "error",
            "message": _sanitize_error(str(exc), settings),
            "batch_results": batch_results,
        }

    post_update = _plan_remote_data_update({**input_data, "dry_run": True}, context)
    remaining = [
        item
        for item in post_update.get("missing_ranges", [])
        if isinstance(item, dict)
    ]
    all_updated = all(item.get("status") == "updated" for item in batch_results)
    post_update_coverage_status = _coverage_status_from_plan(post_update)
    return {
        **base_payload,
        "status": "updated" if all_updated and not remaining else "PARTIAL_UPDATE",
        "dry_run": False,
        "batch_results": batch_results,
        "post_update_coverage": {
            "coverage_status": post_update_coverage_status,
            "data_update_needed": post_update.get("data_update_needed"),
            "missing_ranges": post_update.get("missing_ranges", []),
            "missing_symbols": post_update.get("missing_symbols", []),
            "stale_symbols": post_update.get("stale_symbols", []),
        },
        "remaining_missing_ranges": remaining,
    }


def _coverage_status_from_plan(plan: dict[str, Any]) -> str:
    explicit = plan.get("coverage_status")
    if explicit:
        return str(explicit)
    status = str(plan.get("status") or "")
    if status == "CALENDAR_VALIDATION_REQUIRED":
        return status
    if bool(plan.get("data_update_needed", False)):
        return "PARTIAL_COVERAGE"
    return "OK"


def _plan_fundamental_data_update(
    input_data: dict[str, Any],
    _context: ToolContext,
) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
    try:
        source, start, end = _parse_request(input_data)
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        ts_code = _normalize_ts_code(input_data.get("ts_code"))
        symbols = _normalize_symbols(input_data.get("symbols"))
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    datasets_used = _existing_fundamental_datasets(lake)
    coverage_status = "NO_DATA" if not datasets_used else "PARTIAL_COVERAGE"
    missing_ranges = [{"start_date": start, "end_date": end}] if coverage_status != "OK" else []
    include_daily_basic = bool(input_data.get("include_daily_basic", True))
    include_financial_statements = bool(input_data.get("include_financial_statements", True))
    include_dividend = bool(input_data.get("include_dividend", True))
    requests = build_fundamental_update_plan(
        TushareClient(token=None),
        start,
        end,
        ts_code=ts_code,
        include_daily_basic=include_daily_basic,
        include_financial_statements=include_financial_statements,
        include_dividend=include_dividend,
    )
    return {
        "status": "planned",
        "category": "fundamentals",
        "source": source,
        "start_date": start,
        "end_date": end,
        "requested_start_date": start,
        "requested_end_date": end,
        "requested_symbols": symbols or ([ts_code] if ts_code else []),
        "datasets_used": datasets_used,
        "coverage_status": coverage_status,
        "data_update_needed": coverage_status != "OK",
        "missing_ranges": missing_ranges,
        "next_repair_tool": "run_fundamental_data_update",
        "requests": requests,
        "metadata": {
            "plan_meaning": "dry_run_only_no_remote_fetch_performed",
            "pit_rule": (
                "financial visible_date <= as_of_date; "
                "daily_basic trade_date <= as_of_date"
            ),
            "requested_symbols_count": len(symbols) if symbols else (1 if ts_code else 0),
        },
    }


def _run_fundamental_data_update(
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
    settings = _get_settings()
    try:
        source, start, end = _parse_request(input_data)
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        ts_code = _normalize_ts_code(input_data.get("ts_code"))
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    if _needs_auto_chunk(start, end, settings) and bool(input_data.get("auto_chunk", False)):
        return _run_chunked_fundamental_data_update(
            input_data,
            context,
            settings=settings,
            start=start,
            end=end,
            ts_code=ts_code,
        )

    try:
        _validate_span(start, end, settings.remote_data_max_days_per_call)
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    if bool(input_data.get("dry_run", False)):
        planned = _plan_fundamental_data_update(input_data, context)
        planned["dry_run"] = True
        return planned
    scope_block = _autonomous_fundamental_live_scope_block(input_data, context, ts_code=ts_code)
    if scope_block is not None:
        return scope_block

    if settings.tushare_token is None and _client_factory is None:
        return {
            "status": "NOT_CONFIGURED",
            "message": "TUSHARE_TOKEN is required for live fundamental data update",
            "next_repair_tool": "run_fundamental_data_update",
        }

    try:
        service = _build_update_service(settings, lake)
        result = service.update_fundamentals(
            start,
            end,
            ts_code=ts_code,
            include_daily_basic=bool(input_data.get("include_daily_basic", True)),
            include_financial_statements=bool(
                input_data.get("include_financial_statements", True)
            ),
            include_dividend=bool(input_data.get("include_dividend", True)),
        ).as_dict()
        post_update = _plan_fundamental_data_update({**input_data, "dry_run": True}, context)
        result.update(
            {
                "category": "fundamentals",
                "dry_run": False,
                "datasets_used": _existing_fundamental_datasets(lake),
                "coverage_status": post_update.get("coverage_status"),
                "data_update_needed": post_update.get("data_update_needed"),
                "missing_ranges": post_update.get("missing_ranges", []),
                "next_repair_tool": "run_fundamental_data_update",
            }
        )
        return result
    except Exception as exc:
        return {"status": "error", "message": _sanitize_error(str(exc), settings)}


def _run_chunked_fundamental_data_update(
    input_data: dict[str, Any],
    context: ToolContext,
    *,
    settings: Settings,
    start: str,
    end: str,
    ts_code: str | None,
) -> dict[str, Any]:
    chunks = _chunk_ranges(start, end, settings.remote_data_max_days_per_call)
    batch_plans = [
        _plan_fundamental_data_update(
            {
                **input_data,
                "start_date": chunk_start,
                "end_date": chunk_end,
                "auto_chunk": False,
                "dry_run": True,
            },
            context,
        )
        for chunk_start, chunk_end in chunks
    ]
    batches = [
        {
            **plan,
            "batch_index": index,
            "start_date": chunk_start,
            "end_date": chunk_end,
        }
        for index, (plan, (chunk_start, chunk_end)) in enumerate(
            zip(batch_plans, chunks, strict=True),
            start=1,
        )
    ]
    execute_plan = bool(input_data.get("execute_plan", False))
    dry_run = bool(input_data.get("dry_run", False))
    base_payload: dict[str, Any] = {
        "status": "planned",
        "category": "fundamentals",
        "source": "tushare",
        "start_date": start,
        "end_date": end,
        "requested_start_date": start,
        "requested_end_date": end,
        "auto_chunk": True,
        "execute_plan": execute_plan,
        "dry_run": dry_run,
        "batches": batches,
        "coverage_status": _worst_coverage_status(
            [str(batch.get("coverage_status", "NO_DATA")) for batch in batches]
        ),
        "missing_ranges": [
            item
            for batch in batches
            for item in batch.get("missing_ranges", [])
            if isinstance(item, dict)
        ],
        "next_repair_tool": "run_fundamental_data_update",
    }
    base_payload["remaining_missing_ranges"] = list(base_payload["missing_ranges"])
    if dry_run or not execute_plan:
        return base_payload
    scope_block = _autonomous_fundamental_live_scope_block(input_data, context, ts_code=ts_code)
    if scope_block is not None:
        return {**base_payload, **scope_block}

    if settings.tushare_token is None and _client_factory is None:
        return {
            **base_payload,
            "status": "NOT_CONFIGURED",
            "message": "TUSHARE_TOKEN is required for live fundamental data update",
        }

    batch_results: list[dict[str, Any]] = []
    try:
        service = _build_update_service(settings, _get_lake_required())
        for chunk_start, chunk_end in chunks:
            batch_results.append(
                service.update_fundamentals(
                    chunk_start,
                    chunk_end,
                    ts_code=ts_code,
                    include_daily_basic=bool(input_data.get("include_daily_basic", True)),
                    include_financial_statements=bool(
                        input_data.get("include_financial_statements", True)
                    ),
                    include_dividend=bool(input_data.get("include_dividend", True)),
                ).as_dict()
            )
    except Exception as exc:
        return {
            **base_payload,
            "status": "error",
            "message": _sanitize_error(str(exc), settings),
            "batch_results": batch_results,
        }

    post_update = _plan_fundamental_data_update({**input_data, "dry_run": True}, context)
    all_updated = all(item.get("status") == "updated" for item in batch_results)
    return {
        **base_payload,
        "status": "updated" if all_updated else "PARTIAL_UPDATE",
        "dry_run": False,
        "batch_results": batch_results,
        "post_update_coverage": {
            "coverage_status": post_update.get("coverage_status"),
            "datasets_used": post_update.get("datasets_used", []),
            "missing_ranges": post_update.get("missing_ranges", []),
        },
        "remaining_missing_ranges": []
        if all_updated
        else post_update.get("missing_ranges", []),
        "datasets_used": _existing_fundamental_datasets(_get_lake_required()),
    }


def _autonomous_fundamental_live_scope_block(
    input_data: dict[str, Any],
    context: ToolContext,
    *,
    ts_code: str | None,
) -> dict[str, Any] | None:
    if not context.requested_by_llm or ts_code:
        return None
    symbols = _normalize_symbols(input_data.get("symbols"))
    return {
        "status": "BLOCKED",
        "reason": "AUTONOMOUS_FUNDAMENTAL_UPDATE_REQUIRES_SECURITY_SCOPE",
        "message": (
            "Autonomous live fundamental updates must be scoped to a single ts_code. "
            "Basket symbols are supported for coverage checks, but live basket fills "
            "would fall back to market-wide Tushare requests."
        ),
        "requested_symbols_count": len(symbols),
        "missing_inputs": ["ts_code"],
        "next_repair_tool": "run_fundamental_data_update",
    }


def _plan_macro_data_update(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
    try:
        source, start, end = _parse_request(input_data)
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    datasets = _normalize_macro_datasets(input_data.get("datasets"))
    unknown = [item for item in datasets if item not in MACRO_DATASETS]
    if unknown:
        return {
            "status": "INVALID_REQUEST",
            "category": "macro",
            "message": f"unknown macro dataset(s): {unknown}",
            "requested_datasets": datasets,
            "known_datasets": sorted(MACRO_DATASETS),
            "next_repair_tool": "run_macro_data_update",
        }
    requested = datasets or sorted(MACRO_DATASETS)
    datasets_used = [
        dataset
        for dataset in requested
        if lake.dataset_path("raw", MACRO_DATASETS[dataset].raw_dataset).exists()
    ]
    coverage_status = "OK" if len(datasets_used) == len(requested) else "NO_DATA"
    if datasets_used and coverage_status != "OK":
        coverage_status = "PARTIAL_COVERAGE"
    missing_ranges = [{"start_date": start, "end_date": end}] if coverage_status != "OK" else []
    return {
        "status": "planned",
        "category": "macro",
        "source": source,
        "start_date": start,
        "end_date": end,
        "requested_start_date": start,
        "requested_end_date": end,
        "requested_datasets": requested,
        "known_datasets": sorted(MACRO_DATASETS),
        "datasets_used": datasets_used,
        "coverage_status": coverage_status,
        "data_update_needed": coverage_status != "OK",
        "missing_ranges": missing_ranges,
        "next_repair_tool": "run_macro_data_update",
        "requests": build_macro_update_plan(
            TushareClient(token=None),
            start,
            end,
            datasets=requested,
        ),
        "metadata": {"plan_meaning": "dry_run_only_no_remote_fetch_performed"},
    }


def _run_macro_data_update(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
    settings = _get_settings()
    try:
        _source, start, end = _parse_request(input_data)
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    if _needs_auto_chunk(start, end, settings) and bool(input_data.get("auto_chunk", False)):
        return _run_chunked_macro_data_update(
            input_data,
            context,
            settings=settings,
            start=start,
            end=end,
        )

    try:
        _validate_span(start, end, settings.remote_data_max_days_per_call)
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    planned = _plan_macro_data_update(input_data, context)
    if planned.get("status") == "INVALID_REQUEST" or bool(input_data.get("dry_run", False)):
        planned["dry_run"] = bool(input_data.get("dry_run", False))
        return planned

    if settings.tushare_token is None and _client_factory is None:
        return {
            "status": "NOT_CONFIGURED",
            "message": "TUSHARE_TOKEN is required for live macro data update",
            "next_repair_tool": "run_macro_data_update",
        }
    try:
        service = _build_update_service(settings, lake)
        result = service.update_macro(
            start,
            end,
            datasets=list(planned.get("requested_datasets", [])),
        ).as_dict()
        post_update = _plan_macro_data_update({**input_data, "dry_run": True}, context)
        result.update(
            {
                "category": "macro",
                "dry_run": False,
                "datasets_used": post_update.get("datasets_used", []),
                "coverage_status": post_update.get("coverage_status"),
                "data_update_needed": post_update.get("data_update_needed"),
                "missing_ranges": post_update.get("missing_ranges", []),
                "next_repair_tool": "run_macro_data_update",
            }
        )
        return result
    except Exception as exc:
        return {"status": "error", "message": _sanitize_error(str(exc), settings)}


def _run_chunked_macro_data_update(
    input_data: dict[str, Any],
    context: ToolContext,
    *,
    settings: Settings,
    start: str,
    end: str,
) -> dict[str, Any]:
    chunks = _chunk_ranges(start, end, settings.remote_data_max_days_per_call)
    batch_plans = [
        _plan_macro_data_update(
            {
                **input_data,
                "start_date": chunk_start,
                "end_date": chunk_end,
                "auto_chunk": False,
                "dry_run": True,
            },
            context,
        )
        for chunk_start, chunk_end in chunks
    ]
    batches = [
        {
            **plan,
            "batch_index": index,
            "start_date": chunk_start,
            "end_date": chunk_end,
        }
        for index, (plan, (chunk_start, chunk_end)) in enumerate(
            zip(batch_plans, chunks, strict=True),
            start=1,
        )
    ]
    execute_plan = bool(input_data.get("execute_plan", False))
    dry_run = bool(input_data.get("dry_run", False))
    base_payload: dict[str, Any] = {
        "status": "planned",
        "category": "macro",
        "source": "tushare",
        "start_date": start,
        "end_date": end,
        "requested_start_date": start,
        "requested_end_date": end,
        "auto_chunk": True,
        "execute_plan": execute_plan,
        "dry_run": dry_run,
        "batches": batches,
        "coverage_status": _worst_coverage_status(
            [str(batch.get("coverage_status", "NO_DATA")) for batch in batches]
        ),
        "missing_ranges": [
            item
            for batch in batches
            for item in batch.get("missing_ranges", [])
            if isinstance(item, dict)
        ],
        "next_repair_tool": "run_macro_data_update",
    }
    base_payload["remaining_missing_ranges"] = list(base_payload["missing_ranges"])
    if dry_run or not execute_plan:
        return base_payload

    if settings.tushare_token is None and _client_factory is None:
        return {
            **base_payload,
            "status": "NOT_CONFIGURED",
            "message": "TUSHARE_TOKEN is required for live macro data update",
        }

    batch_results: list[dict[str, Any]] = []
    try:
        service = _build_update_service(settings, _get_lake_required())
        requested = list(batch_plans[0].get("requested_datasets", [])) if batch_plans else []
        for chunk_start, chunk_end in chunks:
            batch_results.append(
                service.update_macro(
                    chunk_start,
                    chunk_end,
                    datasets=requested,
                ).as_dict()
            )
    except Exception as exc:
        return {
            **base_payload,
            "status": "error",
            "message": _sanitize_error(str(exc), settings),
            "batch_results": batch_results,
        }

    post_update = _plan_macro_data_update({**input_data, "dry_run": True}, context)
    all_updated = all(item.get("status") == "updated" for item in batch_results)
    return {
        **base_payload,
        "status": "updated" if all_updated else "PARTIAL_UPDATE",
        "dry_run": False,
        "batch_results": batch_results,
        "post_update_coverage": {
            "coverage_status": post_update.get("coverage_status"),
            "datasets_used": post_update.get("datasets_used", []),
            "missing_ranges": post_update.get("missing_ranges", []),
        },
        "remaining_missing_ranges": []
        if all_updated
        else post_update.get("missing_ranges", []),
        "datasets_used": post_update.get("datasets_used", []),
    }


def _build_update_service(settings: Settings, lake: DataLake) -> TushareDataUpdateService:
    return TushareDataUpdateService(
        _build_client(settings),
        lake,
        limiter=RequestLimiter(min_interval_seconds=settings.remote_data_min_interval_seconds),
        lock_timeout_seconds=settings.remote_data_lock_timeout_seconds,
        retry_attempts=settings.remote_data_retry_attempts,
        retry_backoff_seconds=settings.remote_data_retry_backoff_seconds,
    )


def _get_lake_required() -> DataLake:
    lake = _get_lake()
    if lake is None:
        raise RuntimeError("data lake not wired")
    return lake


def _parse_request(input_data: dict[str, Any]) -> tuple[str, str, str]:
    source = str(input_data.get("source", "tushare")).lower()
    start = str(input_data.get("start_date") or input_data.get("start") or "")
    end = str(input_data.get("end_date") or input_data.get("end") or "")
    if not start or not end:
        raise ValueError("start_date and end_date are required")
    _parse_date(start)
    _parse_date(end)
    if _parse_date(start) > _parse_date(end):
        raise ValueError("start_date must be on or before end_date")
    return source, start, end


def _validate_span(start: str, end: str, max_days: int) -> None:
    days = (_parse_date(end) - _parse_date(start)).days + 1
    if days > max_days:
        raise ValueError(
            f"requested range has {days} days; remote_data_max_days_per_call={max_days}"
        )


def _needs_auto_chunk(start: str, end: str, settings: Settings) -> bool:
    days = (_parse_date(end) - _parse_date(start)).days + 1
    return days > settings.remote_data_max_days_per_call


def _chunk_ranges(start: str, end: str, max_days: int) -> list[tuple[str, str]]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    chunks: list[tuple[str, str]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end_date)
        chunks.append((_format_date(cursor), _format_date(chunk_end)))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _worst_coverage_status(statuses: list[str]) -> str:
    if not statuses:
        return "NO_DATA"
    if all(status == "OK" for status in statuses):
        return "OK"
    if any(status in {"OK", "PARTIAL_COVERAGE", "PARTIAL"} for status in statuses):
        return "PARTIAL_COVERAGE"
    return "NO_DATA"


def _missing_ranges(
    lake: DataLake,
    expected_dates: list[str],
    *,
    ts_code: str | None,
    symbols: list[str],
    asset_type: str,
) -> list[dict[str, str]]:
    if symbols:
        covered_by_date: dict[str, set[str]] = {}
        requested = set(symbols)
        for dataset in _datasets_for_asset_type(asset_type, scoped=True):
            if not lake.dataset_path("raw", dataset).exists():
                continue
            frame = lake.read_parquet("raw", dataset)
            if "trade_date" not in frame.columns or "ts_code" not in frame.columns:
                continue
            frame = frame[frame["ts_code"].astype(str).isin(requested)]
            for row in frame[["ts_code", "trade_date"]].dropna().itertuples(index=False):
                trade_date = _format_date(row.trade_date)
                covered_by_date.setdefault(trade_date, set()).add(str(row.ts_code))
        missing = [
            item
            for item in expected_dates
            if not requested.issubset(covered_by_date.get(item, set()))
        ]
        return _coalesce_dates(missing)

    covered: set[str] = set()
    for dataset in _datasets_for_asset_type(asset_type, scoped=ts_code is not None):
        if not lake.dataset_path("raw", dataset).exists():
            continue
        frame = lake.read_parquet("raw", dataset)
        if "trade_date" not in frame.columns:
            continue
        if ts_code and "ts_code" in frame.columns:
            frame = frame[frame["ts_code"].astype(str) == ts_code]
        covered.update(_format_date(item) for item in frame["trade_date"].dropna().tolist())
    if not covered:
        return _coalesce_dates(expected_dates)
    missing = [item for item in expected_dates if item not in covered]
    return _coalesce_dates(missing)


def _symbol_coverage(
    lake: DataLake,
    expected_dates: list[str],
    *,
    ts_code: str | None,
    symbols: list[str],
    asset_type: str,
) -> dict[str, Any]:
    requested_symbols = symbols or ([ts_code] if ts_code else [])
    if not requested_symbols:
        missing_dates = _missing_dates(lake, expected_dates, ts_code=None, asset_type=asset_type)
        return {
            "coverage_by_symbol": {},
            "missing_symbols": [],
            "stale_symbols": [],
            "covered_symbols": [],
            "missing_ranges": _coalesce_dates(missing_dates),
            "missing_dates_count": len(missing_dates),
        }

    observed_by_symbol: dict[str, set[str]] = {symbol: set() for symbol in requested_symbols}
    requested = set(requested_symbols)
    for dataset in _datasets_for_asset_type(asset_type, scoped=True):
        if not lake.dataset_path("raw", dataset).exists():
            continue
        frame = lake.read_parquet("raw", dataset)
        if frame.empty or "trade_date" not in frame.columns or "ts_code" not in frame.columns:
            continue
        frame = frame[frame["ts_code"].astype(str).isin(requested)]
        for row in frame[["ts_code", "trade_date"]].dropna().itertuples(index=False):
            observed_by_symbol.setdefault(str(row.ts_code), set()).add(
                _format_date(row.trade_date)
            )

    coverage_by_symbol: dict[str, dict[str, Any]] = {}
    missing_symbols: list[str] = []
    stale_symbols: list[str] = []
    covered_symbols: list[str] = []
    all_missing_dates: set[str] = set()
    expected_set = set(expected_dates)
    for symbol in requested_symbols:
        observed_dates = sorted(observed_by_symbol.get(symbol, set()))
        observed_expected_dates = [item for item in observed_dates if item in expected_set]
        missing_dates = [item for item in expected_dates if item not in observed_dates]
        all_missing_dates.update(missing_dates)
        missing_ranges = _coalesce_dates(missing_dates)
        if not observed_dates:
            missing_symbols.append(symbol)
        elif missing_dates:
            stale_symbols.append(symbol)
        else:
            covered_symbols.append(symbol)
        coverage_by_symbol[symbol] = {
            "actual_data_start": observed_dates[0] if observed_dates else None,
            "actual_data_end": observed_dates[-1] if observed_dates else None,
            "actual_rows": len(observed_expected_dates),
            "missing_ranges": missing_ranges,
            "data_freshness": (
                "covers_expected_trading_dates"
                if not missing_dates
                else "missing_expected_trading_dates"
            ),
        }

    missing_dates_sorted = sorted(all_missing_dates)
    return {
        "coverage_by_symbol": coverage_by_symbol,
        "missing_symbols": missing_symbols,
        "stale_symbols": stale_symbols,
        "covered_symbols": covered_symbols,
        "missing_ranges": _coalesce_dates(missing_dates_sorted),
        "missing_dates_count": len(missing_dates_sorted),
    }


def _missing_dates(
    lake: DataLake,
    expected_dates: list[str],
    *,
    ts_code: str | None,
    asset_type: str,
) -> list[str]:
    covered: set[str] = set()
    for dataset in _datasets_for_asset_type(asset_type, scoped=ts_code is not None):
        if not lake.dataset_path("raw", dataset).exists():
            continue
        frame = lake.read_parquet("raw", dataset)
        if "trade_date" not in frame.columns:
            continue
        if ts_code and "ts_code" in frame.columns:
            frame = frame[frame["ts_code"].astype(str) == ts_code]
        covered.update(_format_date(item) for item in frame["trade_date"].dropna().tolist())
    if not covered:
        return expected_dates
    return [item for item in expected_dates if item not in covered]


def _datasets_for_asset_type(asset_type: str, *, scoped: bool) -> tuple[str, ...]:
    if not scoped:
        return ("tushare_daily", "tushare_fund_daily")
    if asset_type == "stock":
        return ("tushare_daily",)
    if asset_type == "etf":
        return ("tushare_fund_daily",)
    return ("tushare_daily", "tushare_fund_daily")


def _actual_data_coverage(
    lake: DataLake,
    *,
    ts_code: str | None,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    observed: list[str] = []
    requested = set(symbols or [])
    for dataset in ("tushare_daily", "tushare_fund_daily"):
        if not lake.dataset_path("raw", dataset).exists():
            continue
        frame = lake.read_parquet("raw", dataset)
        if frame.empty or "trade_date" not in frame.columns:
            continue
        if ts_code and "ts_code" in frame.columns:
            frame = frame[frame["ts_code"].astype(str) == ts_code]
        if requested and "ts_code" in frame.columns:
            frame = frame[frame["ts_code"].astype(str).isin(requested)]
        if frame.empty:
            continue
        observed.extend(_format_date(item) for item in frame["trade_date"].dropna().tolist())
    if not observed:
        return {"actual_data_start": None, "actual_data_end": None, "actual_rows": 0}
    dates = sorted(observed)
    return {
        "actual_data_start": dates[0],
        "actual_data_end": dates[-1],
        "actual_rows": len(dates),
    }


def _expected_coverage_dates(
    lake: DataLake,
    start: str,
    end: str,
) -> tuple[list[str], str]:
    calendar_path = lake.dataset_path("raw", "tushare_trade_calendar")
    if calendar_path.exists():
        frame = lake.read_parquet("raw", "tushare_trade_calendar")
        if not frame.empty and "cal_date" in frame.columns:
            data = frame.copy()
            if "is_open" in data.columns:
                data = data[data["is_open"].astype(int) == 1]
            start_date = _parse_date(start)
            end_date = _parse_date(end)
            dates = [
                _format_date(item)
                for item in data["cal_date"].dropna().tolist()
                if start_date <= _parse_date(_format_date(item)) <= end_date
            ]
            if dates:
                return sorted(set(dates)), "tushare_trade_calendar"
    observed_dates = _observed_market_dates(lake, start, end)
    if observed_dates:
        return observed_dates, "observed_market_daily_dates"
    return [
        current.strftime("%Y%m%d")
        for current in _date_range(_parse_date(start), _parse_date(end))
    ], "calendar_days"


def _observed_market_dates(lake: DataLake, start: str, end: str) -> list[str]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    dates: set[str] = set()
    for dataset in ("tushare_daily", "tushare_fund_daily"):
        if not lake.dataset_path("raw", dataset).exists():
            continue
        frame = lake.read_parquet("raw", dataset)
        if frame.empty or "trade_date" not in frame.columns:
            continue
        for item in frame["trade_date"].dropna().tolist():
            text = _format_date(item)
            parsed = _parse_date(text)
            if start_date <= parsed <= end_date:
                dates.add(text)
    return sorted(dates)


def _coalesce_dates(dates: list[str]) -> list[dict[str, str]]:
    if not dates:
        return []
    ranges: list[dict[str, str]] = []
    range_start = dates[0]
    previous = dates[0]
    for item in dates[1:]:
        if _parse_date(item) != _parse_date(previous) + timedelta(days=1):
            ranges.append({"start_date": range_start, "end_date": previous})
            range_start = item
        previous = item
    ranges.append({"start_date": range_start, "end_date": previous})
    return ranges


def _estimate_request_count(
    input_data: dict[str, Any],
    *,
    missing_dates_count: int,
    data_update_needed: bool,
    scoped: bool,
) -> int:
    include_daily = bool(input_data.get("include_daily", True))
    include_basics = bool(input_data.get("include_basics", True))
    if not data_update_needed:
        return 0
    count = 1
    if include_basics:
        count += 3
    if include_daily:
        count += 1 if scoped else missing_dates_count or 1
        if not scoped:
            count += 1
            count += missing_dates_count
    return count


def _remote_data_update_timeout_seconds(
    input_data: dict[str, Any],
    context: ToolContext,
) -> int:
    settings = _get_settings()
    planned = _plan_remote_data_update({**input_data, "dry_run": True}, context)
    request_count = int(planned.get("estimated_request_count") or 0)
    computed = (
        settings.remote_data_tool_base_timeout_seconds
        + request_count * settings.remote_data_tool_timeout_seconds_per_request
    )
    return min(
        max(300, computed),
        settings.remote_data_tool_max_timeout_seconds,
    )


def _estimated_request_count(planned: dict[str, Any]) -> int:
    try:
        return int(planned.get("estimated_request_count") or 0)
    except (TypeError, ValueError):
        return 0


def _next_remote_repair_tool(input_data: dict[str, Any]) -> str:
    include_basics = bool(input_data.get("include_basics", True))
    include_daily = bool(input_data.get("include_daily", True))
    if include_basics and not include_daily:
        return "run_fundamental_data_update"
    return "run_remote_data_update"


def _copy_coverage_fields(payload: dict[str, Any], plan: dict[str, Any]) -> None:
    for key in (
        "coverage_by_symbol",
        "missing_symbols",
        "stale_symbols",
        "covered_symbols",
        "missing_ranges",
        "estimated_request_count",
        "data_freshness",
    ):
        if key in plan:
            payload[key] = plan[key]


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _parse_date(value: str) -> date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"invalid date: {value}")


def _format_date(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")  # type: ignore[no-any-return]
    text = str(value)
    if "-" in text:
        return datetime.fromisoformat(text).strftime("%Y%m%d")
    return text


def _sanitize_error(message: str, settings: Settings) -> str:
    if settings.tushare_token is not None:
        token = settings.tushare_token.get_secret_value()
        if token:
            message = message.replace(token, "[redacted]")
    return message


def _existing_fundamental_datasets(lake: DataLake) -> list[str]:
    candidates = ["tushare_daily_basic", *(item[0] for item in FINANCIAL_TABLES.values())]
    return [dataset for dataset in candidates if lake.dataset_path("raw", dataset).exists()]


def _normalize_macro_datasets(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    normalized: list[str] = []
    for item in raw_items:
        text = item.strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


_UPDATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Only tushare is currently supported."},
        "start_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "end_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "include_daily": {"type": "boolean"},
        "include_basics": {"type": "boolean"},
        "ts_code": {"type": "string", "description": "Optional security code, e.g. 159259.SZ."},
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional basket symbols for coverage checks. Live stock updates still "
                "use batch or market-wide requests instead of per-symbol fan-out."
            ),
        },
        "asset_type": {
            "type": "string",
            "description": (
                "stock, etf, or auto. auto uses local/remote fund_basic when "
                "ts_code is an ETF."
            ),
        },
        "dry_run": {
            "type": "boolean",
            "description": "When true, return the local gap plan without contacting Tushare.",
        },
        "auto_chunk": {
            "type": "boolean",
            "description": (
                "When true, split requests larger than remote_data_max_days_per_call "
                "into legal date batches."
            ),
        },
        "execute_plan": {
            "type": "boolean",
            "description": (
                "When true with auto_chunk=true and dry_run=false, execute each planned "
                "batch and return post_update_coverage."
            ),
        },
    },
    "required": ["start_date", "end_date"],
    "additionalProperties": False,
}


def _normalize_ts_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "." not in text and text.isdigit() and len(text) == 6:
        text = f"{text}.SZ" if text.startswith(("0", "1", "2", "3")) else f"{text}.SH"
    return text


def _normalize_symbols(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = _normalize_ts_code(item)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _effective_start_from_local_basics(
    lake: DataLake,
    start: str,
    *,
    ts_code: str | None,
    asset_type: str,
) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {"asset_type": asset_type}
    if not ts_code or asset_type not in {"auto", "etf"}:
        return start, metadata
    metadata["ts_code"] = ts_code
    if not lake.dataset_path("raw", "tushare_etf_basic").exists():
        return start, metadata
    frame = lake.read_parquet("raw", "tushare_etf_basic")
    if frame.empty or "ts_code" not in frame.columns or "list_date" not in frame.columns:
        return start, metadata
    matches = frame[frame["ts_code"].astype(str) == ts_code]
    if matches.empty:
        return start, metadata
    list_date = str(matches.iloc[0]["list_date"])
    metadata["asset_type"] = "etf"
    metadata["list_date"] = list_date
    if list_date > start:
        metadata["requested_start"] = start
        metadata["start_adjusted"] = True
        return list_date, metadata
    metadata["start_adjusted"] = False
    return start, metadata


run_remote_data_update_tool: AgentTool = tool(
    ToolSpec(
        name="run_remote_data_update",
        description="通过本地受控同步器补齐 Tushare 远程数据，强制限速、锁和日期跨度上限。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_formal",
        input_schema=_UPDATE_INPUT_SCHEMA,
        output_schema={"type": "object"},
        deterministic=False,
        timeout_seconds=300,
    ),
    fn=_run_remote_data_update,
    timeout_seconds_for_call=_remote_data_update_timeout_seconds,
)


_FUNDAMENTAL_UPDATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Only tushare is currently supported."},
        "start_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "end_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "ts_code": {"type": "string", "description": "Optional security code."},
        "symbols": {"type": "array", "items": {"type": "string"}},
        "include_daily_basic": {"type": "boolean"},
        "include_financial_statements": {"type": "boolean"},
        "include_dividend": {"type": "boolean"},
        "dry_run": {
            "type": "boolean",
            "description": "When true, return the local gap plan without contacting Tushare.",
        },
        "auto_chunk": {
            "type": "boolean",
            "description": (
                "When true, split requests larger than remote_data_max_days_per_call "
                "into legal date batches."
            ),
        },
        "execute_plan": {
            "type": "boolean",
            "description": (
                "When true with auto_chunk=true and dry_run=false, execute each planned "
                "batch and return post_update_coverage."
            ),
        },
    },
    "required": ["start_date", "end_date"],
    "additionalProperties": False,
}


_MACRO_UPDATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Only tushare is currently supported."},
        "start_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "end_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "datasets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Macro dataset ids such as cn_cpi, cn_ppi, cn_gdp, or shibor.",
        },
        "dry_run": {
            "type": "boolean",
            "description": "When true, return the local gap plan without contacting Tushare.",
        },
    },
    "required": ["start_date", "end_date"],
    "additionalProperties": False,
}


run_fundamental_data_update_tool: AgentTool = tool(
    ToolSpec(
        name="run_fundamental_data_update",
        description=(
            "通过本地受控同步器补齐 Tushare 基本面与日频估值数据。"
            "先用 dry_run=true, auto_chunk=true 查看缺口和批次计划；"
            "允许 live update 时用 dry_run=false, auto_chunk=true, execute_plan=true "
            "逐批执行并返回 post_update_coverage。"
        ),
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_formal",
        input_schema=_FUNDAMENTAL_UPDATE_INPUT_SCHEMA,
        output_schema={"type": "object"},
        deterministic=False,
        timeout_seconds=300,
    ),
    fn=_run_fundamental_data_update,
)


run_macro_data_update_tool: AgentTool = tool(
    ToolSpec(
        name="run_macro_data_update",
        description=(
            "通过本地受控同步器补齐 Tushare 宏观数据集。"
            "可用 datasets 指定 cn_cpi、cn_ppi、cn_gdp、shibor。"
            "超范围请求应使用 auto_chunk=true 拆窗；允许 live update 时设置 "
            "execute_plan=true 后复查 post_update_coverage。"
        ),
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_formal",
        input_schema=_MACRO_UPDATE_INPUT_SCHEMA,
        output_schema={"type": "object"},
        deterministic=False,
        timeout_seconds=300,
    ),
    fn=_run_macro_data_update,
)


def build_remote_data_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    return [
        tool(
            run_remote_data_update_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _run_remote_data_update, input_data, context
            ),
            timeout_seconds_for_call=lambda input_data, context: _timeout_with_deps(
                deps,
                input_data,
                context,
            ),
        ),
        tool(
            run_fundamental_data_update_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps,
                _run_fundamental_data_update,
                input_data,
                context,
            ),
        ),
        tool(
            run_macro_data_update_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps,
                _run_macro_data_update,
                input_data,
                context,
            ),
        ),
    ]
