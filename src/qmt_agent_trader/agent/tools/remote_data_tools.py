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
    return TushareClient(
        token=token,
        timeout_seconds=settings.remote_data_http_timeout_seconds,
    )


def _plan_remote_data_update(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
    try:
        source, start, end = _parse_request(input_data)
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        ts_code = _normalize_ts_code(input_data.get("ts_code"))
        asset_type = str(input_data.get("asset_type", "stock")).lower()
        effective_start, metadata = _effective_start_from_local_basics(
            lake, start, ts_code=ts_code, asset_type=asset_type
        )
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
        missing_ranges = _missing_ranges(lake, expected_dates)
        metadata["plan_meaning"] = "dry_run_only_no_remote_fetch_performed"
        metadata["calendar_source"] = calendar_source
        metadata["missing_ranges_are_calendar_days"] = not uses_date_calendar
        metadata["requires_trade_calendar_validation"] = not uses_date_calendar
        if not uses_date_calendar:
            metadata["warning"] = (
                "missing_ranges are calendar-day gaps; do not claim they are "
                "weekends or holidays without trade-calendar validation"
            )
        actual_coverage = _actual_data_coverage(lake, ts_code=ts_code)
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
            "requests": build_data_update_plan(TushareClient(token=None), effective_start, end),
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
        ts_code = _normalize_ts_code(input_data.get("ts_code"))
        asset_type = str(input_data.get("asset_type", "stock")).lower()
        if source != "tushare":
            return {"status": "INVALID_REQUEST", "message": "only tushare is supported"}
        if asset_type not in {"stock", "etf", "auto"}:
            return {
                "status": "INVALID_REQUEST",
                "message": "asset_type must be stock, etf, or auto",
            }
        if ts_code is None:
            _validate_span(start, end, settings.remote_data_max_days_per_call)
    except ValueError as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}

    if bool(input_data.get("dry_run", False)):
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
        )
        payload = result.as_dict()
        actual_coverage = _actual_data_coverage(lake, ts_code=ts_code)
        payload.update(
            {
                "requested_start_date": start,
                "requested_end_date": end,
                "actual_data_start": actual_coverage.get("actual_data_start"),
                "actual_data_end": actual_coverage.get("actual_data_end"),
                "coverage_start_date": actual_coverage.get("actual_data_start"),
                "coverage_end_date": actual_coverage.get("actual_data_end"),
            }
        )
        metadata = payload.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.update(actual_coverage)
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


def _missing_ranges(lake: DataLake, expected_dates: list[str]) -> list[dict[str, str]]:
    covered: set[str] = set()
    for dataset in ("tushare_daily", "tushare_fund_daily"):
        if not lake.dataset_path("raw", dataset).exists():
            continue
        frame = lake.read_parquet("raw", dataset)
        if "trade_date" not in frame.columns:
            continue
        covered.update(_format_date(item) for item in frame["trade_date"].dropna().tolist())
    if not covered:
        return _coalesce_dates(expected_dates)
    missing = [item for item in expected_dates if item not in covered]
    return _coalesce_dates(missing)


def _actual_data_coverage(lake: DataLake, *, ts_code: str | None) -> dict[str, Any]:
    observed: list[str] = []
    for dataset in ("tushare_daily", "tushare_fund_daily"):
        if not lake.dataset_path("raw", dataset).exists():
            continue
        frame = lake.read_parquet("raw", dataset)
        if frame.empty or "trade_date" not in frame.columns:
            continue
        if ts_code and "ts_code" in frame.columns:
            frame = frame[frame["ts_code"].astype(str) == ts_code]
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
        "ts_code": {"type": "string", "description": "Optional security code, e.g. 159259.SZ."},
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
)
