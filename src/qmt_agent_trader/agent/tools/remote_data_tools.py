"""Controlled remote data update tools for the research agent."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient
from qmt_agent_trader.services.data_update_service import (
    RequestLimiter,
    TushareDataUpdateService,
    build_data_update_plan,
)

_lake: DataLake | None = None
_settings: Settings | None = None
_client_factory: Callable[[], TushareClient] | None = None


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
    return _lake


def _get_settings() -> Settings:
    return _settings or get_settings()


def _build_client(settings: Settings) -> TushareClient:
    if _client_factory is not None:
        return _client_factory()
    token = settings.tushare_token.get_secret_value() if settings.tushare_token else None
    return TushareClient(token=token)


def _plan_remote_data_update(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
    try:
        source, start, end = _parse_request(input_data)
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        missing_ranges = _missing_ranges(lake, start, end)
        return {
            "status": "planned",
            "source": source,
            "start_date": start,
            "end_date": end,
            "missing_ranges": missing_ranges,
            "requests": build_data_update_plan(TushareClient(token=None), start, end),
        }
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}


def _run_remote_data_update(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}

    settings = _get_settings()
    try:
        source, start, end = _parse_request(input_data)
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        _validate_span(start, end, settings.remote_data_max_days_per_call)
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    if context.dry_run:
        planned = _plan_remote_data_update(input_data, context)
        planned["dry_run"] = True
        return planned

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
        )
        result = service.update(
            start,
            end,
            include_daily=bool(input_data.get("include_daily", True)),
            include_basics=bool(input_data.get("include_basics", True)),
        )
        return result.as_dict()
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


def _missing_ranges(lake: DataLake, start: str, end: str) -> list[dict[str, str]]:
    if not lake.dataset_path("raw", "tushare_daily").exists():
        return [{"start_date": start, "end_date": end}]
    frame = lake.read_parquet("raw", "tushare_daily")
    if "trade_date" not in frame.columns:
        return [{"start_date": start, "end_date": end}]
    covered = {_format_date(item) for item in frame["trade_date"].dropna().tolist()}
    missing = [
        current.strftime("%Y%m%d")
        for current in _date_range(_parse_date(start), _parse_date(end))
        if current.strftime("%Y%m%d") not in covered
    ]
    return _coalesce_dates(missing)


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


_UPDATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Only tushare is currently supported."},
        "start_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "end_date": {"type": "string", "description": "YYYYMMDD or YYYY-MM-DD."},
        "include_daily": {"type": "boolean"},
        "include_basics": {"type": "boolean"},
    },
    "required": ["start_date", "end_date"],
    "additionalProperties": False,
}


plan_remote_data_update_tool: AgentTool = tool(
    ToolSpec(
        name="plan_remote_data_update",
        description="规划远程数据补齐请求，只检查本地覆盖缺口，不联网不写入。",
        permission=PermissionLevel.READ_ONLY,
        input_schema=_UPDATE_INPUT_SCHEMA,
        output_schema={"type": "object"},
        deterministic=False,
    ),
    fn=_plan_remote_data_update,
)


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
)
