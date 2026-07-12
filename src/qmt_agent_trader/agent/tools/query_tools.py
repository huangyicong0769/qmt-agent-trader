"""Data tools: list_data_catalog, query_universe, query_bars, query_fundamentals_pit.

These tools are stubs when the underlying data layer is unavailable — they
return `NOT_AVAILABLE` rather than crashing the Agent loop.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import date, datetime
from typing import Any

import pandas as pd

import qmt_agent_trader.agent.tools.universe_tools as _universe_tools
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    ExecutionStatus,
    RecommendationStatus,
)
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import SHANGHAI_TZ
from qmt_agent_trader.data.bars import (
    enrich_trade_states,
    normalize_tushare_daily,
)
from qmt_agent_trader.data.catalog import visible_dataset_names
from qmt_agent_trader.data.field_sources import (
    FieldSourceIndex,
    FieldSourceSpec,
    FillPolicy,
    fetch_columns_for_source,
)
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.fundamentals import (
    DAILY_BASIC_DATASET,
    DEFAULT_FUNDAMENTAL_FIELDS,
    FINANCIAL_DATASETS,
    load_fundamentals_asof,
    records_jsonable,
)
from qmt_agent_trader.data.macro import MACRO_DATASETS
from qmt_agent_trader.data.providers.tushare.registry import default_tushare_registry
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.transforms.macro_pit import (
    load_macro_series_asof,
)
from qmt_agent_trader.data.transforms.macro_pit import (
    records_jsonable as macro_records_jsonable,
)

MAX_FUNDAMENTAL_ROWS = 500
MAX_MACRO_ROWS = 500
DEFAULT_BAR_LIMIT = 2000
MAX_BAR_LIMIT = 10000

_lake: DataLake | None = None
_lake_var: ContextVar[DataLake | None] = ContextVar("query_tool_lake", default=None)


def set_data_lake(lake: DataLake) -> None:
    global _lake
    _lake = lake
    _universe_tools.set_data_lake(lake)


def _get_lake() -> DataLake | None:
    return _lake_var.get() or _lake


def _with_deps(
    deps: AgentToolDependencies,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    token = _lake_var.set(deps.data_lake)
    try:
        return fn(input_data, context)
    finally:
        _lake_var.reset(token)


# ── list_data_catalog ────────────────────────────────────────────────────────


def _list_data_catalog(_input: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return _with_query_evidence_status(
            {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
        )
    try:
        layers = {
            layer: visible_dataset_names(layer, lake.list_dataset_names(layer))
            for layer in ("raw", "bronze", "silver", "gold")
        }
        return {"status": "ok", "layers": layers}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


list_data_catalog_tool: AgentTool = tool(
    ToolSpec(
        name="list_data_catalog",
        description="查看当前有哪些数据表、字段、日期范围和覆盖率。",
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_list_data_catalog,
)

# ── query_universe ───────────────────────────────────────────────────────────


query_universe_tool = _universe_tools.query_universe_tool


def _date_key(value: str) -> str:
    return _parse_date(value).strftime("%Y%m%d")


def _with_query_evidence_status(payload: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = payload.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    status = str(payload.get("status") or metadata.get("status") or "UNKNOWN")
    enriched = dict(payload)
    enriched["status"] = status
    enriched["execution_status"] = ExecutionStatus.OK.value
    enriched["raw_status"] = payload.get("status") or metadata.get("status")
    enriched["message"] = payload.get("message") or metadata.get("message")
    enriched["reason"] = payload.get("reason") or metadata.get("reason")
    enriched["next_repair_tool"] = payload.get("next_repair_tool") or metadata.get(
        "next_repair_tool"
    )
    enriched["suggested_repair"] = payload.get("suggested_repair") or metadata.get(
        "suggested_repair"
    )
    enriched["repair_action"] = payload.get("repair_action") or metadata.get("repair_action")
    enriched["verification_action"] = payload.get("verification_action") or metadata.get(
        "verification_action"
    )
    enriched["coverage_status"] = (
        payload.get("coverage_status")
        or metadata.get("coverage_status")
        or _coverage_status_for_query_status(status)
    )
    warnings: list[str] = []
    for key in ("warning", "warnings"):
        value = payload.get(key) or metadata.get(key)
        if isinstance(value, list):
            warnings.extend(str(item) for item in value)
        elif value:
            warnings.append(str(value))
    if status in {"OK", "ok"}:
        domain = DomainStatus.OK.value
        evidence = EvidenceStatus.VALID.value
        recommendation = RecommendationStatus.RESEARCH_ONLY.value
    elif status in {"PARTIAL_COVERAGE", "PARTIAL", "PIT_NOT_VALIDATED"}:
        domain = DomainStatus.PARTIAL.value
        evidence = EvidenceStatus.INCOMPLETE.value
        recommendation = RecommendationStatus.UNKNOWN.value
    elif status in {"NO_DATA", "NO_MATCHING_BARS"}:
        domain = DomainStatus.NO_DATA.value
        evidence = EvidenceStatus.INCOMPLETE.value
        recommendation = RecommendationStatus.BLOCKED.value
    elif status in {"INVALID_REQUEST"}:
        domain = DomainStatus.INVALID_REQUEST.value
        evidence = EvidenceStatus.INVALID.value
        recommendation = RecommendationStatus.BLOCKED.value
    elif status in {"NOT_AVAILABLE", "NOT_CONFIGURED"}:
        domain = DomainStatus.NOT_CONFIGURED.value
        evidence = EvidenceStatus.BLOCKED.value
        recommendation = RecommendationStatus.BLOCKED.value
    elif status in {"ERROR"} or metadata.get("error"):
        domain = DomainStatus.FAILED.value
        evidence = EvidenceStatus.INVALID.value
        recommendation = RecommendationStatus.BLOCKED.value
    else:
        domain = DomainStatus.UNKNOWN.value
        evidence = EvidenceStatus.UNKNOWN.value
        recommendation = RecommendationStatus.UNKNOWN.value
        warnings.append("legacy_unstructured_tool_result")
    enriched["domain_status"] = domain
    enriched["evidence_status"] = evidence
    enriched["recommendation_status"] = recommendation
    enriched["warnings"] = sorted(set(warnings))
    blockers: list[str] = []
    if domain in {
        DomainStatus.BLOCKED.value,
        DomainStatus.FAILED.value,
        DomainStatus.NO_DATA.value,
        DomainStatus.INVALID_REQUEST.value,
        DomainStatus.NOT_CONFIGURED.value,
    }:
        reason = enriched.get("reason") or enriched.get("message") or status
        blockers.append(str(reason))
    enriched["blockers"] = blockers
    return enriched


def _coverage_status_for_query_status(status: str) -> str:
    if status in {"OK", "ok"}:
        return "OK"
    if status in {"PARTIAL_COVERAGE", "PARTIAL", "PIT_NOT_VALIDATED"}:
        return "PARTIAL_COVERAGE"
    if status in {"NO_DATA", "NO_MATCHING_BARS"}:
        return "NO_DATA"
    if status == "INVALID_REQUEST":
        return "INVALID_REQUEST"
    if status in {"NOT_AVAILABLE", "NOT_CONFIGURED"}:
        return "BLOCKED"
    if status == "ERROR":
        return "INVALID_REQUEST"
    return "UNKNOWN"

# ── query_bars ───────────────────────────────────────────────────────────────


def _query_bars(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}

    symbols = _requested_symbols(input_data)
    start = input_data.get("start_date", "20200101")
    end = input_data.get("end_date", _today_yyyymmdd())
    requested_fields = input_data.get(
        "fields",
        ["symbol", "trade_date", "open", "high", "low", "close", "volume"],
    )
    fields = _bar_output_fields(requested_fields)
    limit = _bar_limit(input_data.get("limit", DEFAULT_BAR_LIMIT))
    if limit is None:
        return _with_query_evidence_status(
            {
                "rows": [],
                "metadata": {
                    "status": "INVALID_REQUEST",
                    "message": f"limit must be between 1 and {MAX_BAR_LIMIT}",
                },
            }
        )
    include_trade_state = bool(input_data.get("include_trade_state", True))
    if "enrich" in input_data:
        include_trade_state = bool(input_data["enrich"])
    order = str(input_data.get("order", "asc")).lower()
    if order not in {"asc", "desc"}:
        return _with_query_evidence_status(
            {
                "rows": [],
                "metadata": {
                    "status": "INVALID_REQUEST",
                    "message": "order must be asc or desc",
                },
            }
        )

    try:
        bars = _load_bars_for_query(
            lake,
            start=start,
            end=end,
            symbols=symbols or None,
            limit=limit,
            include_trade_state=include_trade_state,
            order=order,
        )
        coverage_metadata = (
            _bars_coverage_metadata_from_lake(
                lake,
                symbols=symbols,
                start=str(start),
                end=str(end),
            )
            if symbols
            else _bars_coverage_metadata(symbols, bars, end)
        )
        if bars.empty:
            metadata: dict[str, Any] = {
                "requested_start_date": str(start),
                "requested_end_date": str(end),
                "returned": 0,
                "limit": limit,
                "backend_limited": True,
                "include_trade_state": include_trade_state,
                "read_strategy": "predicate_pushdown",
                **coverage_metadata,
            }
            if symbols:
                metadata["requested_symbols"] = symbols
                metadata["reason"] = "no matching bars"
            else:
                metadata["status"] = "NO_MATCHING_BARS"
            return _with_query_evidence_status({"rows": [], "metadata": metadata})
        cols = [c for c in fields if c in bars.columns]
        output = bars[cols].head(limit).to_dict(orient="records")
        actual_start, actual_end = _coverage_actual_bounds(coverage_metadata)
        return _with_query_evidence_status({
            "rows": output,
            "metadata": {
                "requested_symbols": symbols,
                "requested": len(symbols),
                "requested_start_date": str(start),
                "requested_end_date": str(end),
                "actual_start_date": actual_start or str(bars["trade_date"].min()),
                "actual_end_date": actual_end or str(bars["trade_date"].max()),
                "data_freshness": (
                    "covers_requested_end"
                    if coverage_metadata.get("status") == "OK"
                    else "stale_vs_requested_end"
                ),
                "returned": len(output),
                "total_rows": len(bars),
                "limit": limit,
                "backend_limited": True,
                "include_trade_state": include_trade_state,
                "read_strategy": "predicate_pushdown",
                "identity_fields_forced": True,
                **coverage_metadata,
            },
        })
    except Exception as exc:
        return _with_query_evidence_status(
            {"rows": [], "metadata": {"status": "ERROR", "error": str(exc)}}
        )


def _bar_output_fields(requested_fields: Any) -> list[str]:
    raw_fields = requested_fields if isinstance(requested_fields, list) else []
    fields: list[str] = []
    for field in ["symbol", "trade_date", *[str(field) for field in raw_fields]]:
        if field not in fields:
            fields.append(field)
    return fields


def _bar_limit(value: Any) -> int | None:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    if limit < 1 or limit > MAX_BAR_LIMIT:
        return None
    return limit


def _load_bars_for_query(
    lake: DataLake,
    *,
    start: str,
    end: str,
    symbols: list[str] | None,
    limit: int,
    include_trade_state: bool,
    order: str,
) -> pd.DataFrame:
    raw = _read_limited_raw_bars(
        lake,
        start=start,
        end=end,
        symbols=symbols,
        limit=limit,
        order=order,
    )
    bars = normalize_tushare_daily(raw)
    if bars.empty:
        return bars
    ascending = order == "asc"
    bars = bars.sort_values(["trade_date", "symbol"], ascending=[ascending, True]).head(limit)
    if not include_trade_state:
        return bars.reset_index(drop=True)

    returned_symbols = sorted(bars["symbol"].astype(str).unique())
    returned_start = min(bars["trade_date"])
    returned_end = max(bars["trade_date"])
    return enrich_trade_states(
        bars,
        suspend=lake.read_parquet_filtered(
            "raw",
            "tushare/suspend_d",
            columns=["ts_code", "trade_date", "suspend_type"],
            start=returned_start,
            end=returned_end,
            symbols=returned_symbols,
        ),
        stk_limit=lake.read_parquet_filtered(
            "raw",
            "tushare/stk_limit",
            columns=["ts_code", "trade_date", "up_limit", "down_limit"],
            start=returned_start,
            end=returned_end,
            symbols=returned_symbols,
        ),
        namechange=lake.read_parquet_filtered(
            "raw",
            "tushare/namechange",
            columns=["ts_code", "name", "start_date", "end_date"],
            symbols=returned_symbols,
        ),
        stock_basic=lake.read_parquet_filtered(
            "raw",
            "tushare/stock_basic",
            columns=["ts_code", "name"],
            symbols=returned_symbols,
        ),
    ).reset_index(drop=True)


def _read_limited_raw_bars(
    lake: DataLake,
    *,
    start: str,
    end: str,
    symbols: list[str] | None,
    limit: int,
    order: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in ("tushare/daily", "tushare/fund_daily"):
        path = lake.dataset_path("raw", dataset)
        if not path.exists():
            continue
        escaped_path = str(path).replace("'", "''")
        columns = _available_bar_columns(lake, escaped_path)
        if not {"ts_code", "trade_date", "open", "high", "low", "close"}.issubset(columns):
            continue
        selected = [
            column
            for column in [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "volume",
                "amount",
                "turnover",
            ]
            if column in columns
        ]
        predicates = [
            _bar_date_key_sql("trade_date") + " >= $start_date",
            _bar_date_key_sql("trade_date") + " <= $end_date",
        ]
        params: dict[str, Any] = {
            "start_date": _date_key(str(start)),
            "end_date": _date_key(str(end)),
        }
        if symbols:
            symbol_values = ", ".join(_sql_string_literal(symbol) for symbol in symbols)
            predicates.append(f"ts_code IN ({symbol_values})")
        sort_direction = "ASC" if order == "asc" else "DESC"
        sql = f"""
            SELECT {", ".join(selected)}
            FROM read_parquet('{escaped_path}')
            WHERE {" AND ".join(predicates)}
            ORDER BY {_bar_date_key_sql("trade_date")} {sort_direction}, ts_code ASC
            LIMIT {int(limit)}
        """
        frame = lake.query_external(sql, params)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _available_bar_columns(lake: DataLake, escaped_path: str) -> set[str]:
    schema = lake.query_external(f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')")
    return {str(item) for item in schema["column_name"].tolist()}


def _bar_date_key_sql(column: str) -> str:
    return (
        "COALESCE("
        f"strftime(try_strptime(CAST({column} AS VARCHAR), '%Y%m%d'), '%Y%m%d'), "
        f"strftime(TRY_CAST({column} AS DATE), '%Y%m%d'), "
        f"substr(regexp_replace(CAST({column} AS VARCHAR), '[^0-9]', '', 'g'), 1, 8)"
        ")"
    )


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _requested_symbols(input_data: dict[str, Any]) -> list[str]:
    raw_symbols: list[Any] = []
    symbols_value = input_data.get("symbols", [])
    if isinstance(symbols_value, list):
        raw_symbols.extend(symbols_value)
    elif symbols_value:
        raw_symbols.append(symbols_value)
    for alias in ("symbol", "code"):
        if input_data.get(alias):
            raw_symbols.append(input_data[alias])

    normalized: list[str] = []
    for raw in raw_symbols:
        text = str(raw).strip()
        if not text:
            continue
        if "." not in text and text.isdigit() and len(text) == 6:
            text = f"{text}.SZ" if text.startswith(("0", "1", "2", "3")) else f"{text}.SH"
        if text not in normalized:
            normalized.append(text)
    return normalized


def _bars_coverage_metadata(symbols: list[str], bars: Any, requested_end: str) -> dict[str, Any]:
    if not symbols:
        return {
            "status": "OK" if not bars.empty else "NO_MATCHING_BARS",
            "coverage_by_symbol": {},
            "missing_symbols": [],
            "stale_symbols": [],
            "covered_symbols": [],
        }

    requested_end_date = _parse_date(str(requested_end))
    coverage_by_symbol: dict[str, dict[str, Any]] = {}
    missing_symbols: list[str] = []
    stale_symbols: list[str] = []
    covered_symbols: list[str] = []

    for symbol in symbols:
        symbol_bars = bars[bars["symbol"] == symbol] if not bars.empty else bars
        returned = len(symbol_bars)
        if returned == 0:
            missing_symbols.append(symbol)
            coverage_by_symbol[symbol] = {
                "returned": 0,
                "actual_start_date": None,
                "actual_end_date": None,
                "data_freshness": "missing_expected_trading_dates",
            }
            continue

        actual_start = symbol_bars["trade_date"].min()
        actual_end = symbol_bars["trade_date"].max()
        actual_end_date = _parse_date(str(actual_end))
        data_freshness = _freshness(str(actual_end), str(requested_end))
        if actual_end_date < requested_end_date:
            stale_symbols.append(symbol)
        else:
            covered_symbols.append(symbol)
        coverage_by_symbol[symbol] = {
            "returned": returned,
            "actual_start_date": _parse_date(str(actual_start)).isoformat(),
            "actual_end_date": _parse_date(str(actual_end)).isoformat(),
            "data_freshness": data_freshness,
        }

    if len(missing_symbols) == len(symbols):
        status = "NO_MATCHING_BARS"
    elif missing_symbols or stale_symbols:
        status = "PARTIAL_COVERAGE"
    else:
        status = "OK"
    payload: dict[str, Any] = {
        "status": status,
        "coverage_by_symbol": coverage_by_symbol,
        "missing_symbols": missing_symbols,
        "stale_symbols": stale_symbols,
        "covered_symbols": covered_symbols,
    }
    if status != "OK":
        payload["next_repair_tool"] = "run_tushare_fetch"
    return payload


def _bars_coverage_metadata_from_lake(
    lake: DataLake,
    *,
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, Any]:
    if not symbols:
        return _bars_coverage_metadata(symbols, pd.DataFrame(), end)
    frames: list[pd.DataFrame] = []
    for dataset in ("tushare/daily", "tushare/fund_daily"):
        frame = lake.read_parquet_filtered(
            "raw",
            dataset,
            columns=["ts_code", "trade_date"],
            start=start,
            end=end,
            symbols=symbols,
        )
        if not frame.empty:
            frames.append(frame.rename(columns={"ts_code": "symbol"}))
    bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _bars_coverage_metadata(symbols, bars, end)


def _coverage_actual_bounds(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    coverage = metadata.get("coverage_by_symbol")
    if not isinstance(coverage, dict):
        return None, None
    starts: list[str] = []
    ends: list[str] = []
    for item in coverage.values():
        if not isinstance(item, dict):
            continue
        start = item.get("actual_start_date")
        end = item.get("actual_end_date")
        if start:
            starts.append(str(start))
        if end:
            ends.append(str(end))
    if not starts or not ends:
        return None, None
    return _parse_date(min(starts)).isoformat(), _parse_date(max(ends)).isoformat()


def _today_yyyymmdd() -> str:
    return datetime.now(tz=SHANGHAI_TZ).strftime("%Y%m%d")


def _freshness(actual_end: str, requested_end: str) -> str:
    return (
        "stale_vs_requested_end"
        if datetime.fromisoformat(actual_end).date() < _parse_date(str(requested_end))
        else "covers_requested_end"
    )


def _parse_date(value: str) -> date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(value).date()


query_bars_tool: AgentTool = tool(
    ToolSpec(
        name="query_bars",
        description="查询历史行情数据。",
        permission=PermissionLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}},
                "symbol": {"type": "string"},
                "code": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
        },
        deterministic=False,
    ),
    fn=_query_bars,
)

# ── query_fundamentals_pit ───────────────────────────────────────────────────


def _query_fundamentals_pit(
    input_data: dict[str, Any], _context: ToolContext
) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return _with_query_evidence_status(
            {
                "rows": [],
                "metadata": {
                    "point_in_time": True,
                    "status": "NOT_AVAILABLE",
                    "message": "data lake not wired",
                },
            }
        )

    as_of = input_data.get("as_of_date")
    if not as_of:
        return _with_query_evidence_status(
            {
                "rows": [],
                "metadata": {
                    "point_in_time": True,
                    "status": "INVALID_REQUEST",
                    "message": "as_of_date is required",
                },
            }
        )
    symbols = _requested_symbols(input_data)
    requested_fields = input_data.get("fields")
    fields = (
        [str(field) for field in requested_fields]
        if isinstance(requested_fields, list)
        else DEFAULT_FUNDAMENTAL_FIELDS
    )
    include_daily_basic = bool(input_data.get("include_daily_basic", True))
    include_financials = bool(input_data.get("include_financials", True))

    datasets_used = _fundamental_datasets_used(lake, include_daily_basic, include_financials)
    if not datasets_used:
        repair_action = _fundamental_repair_action(
            fields=fields,
            symbols=symbols,
            as_of_date=str(as_of),
            missing_symbols=symbols,
            missing_fields={field: symbols for field in fields},
        )
        return _with_query_evidence_status({
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "NO_DATA",
                "as_of_date": str(as_of),
                "requested_symbols": symbols,
                "datasets_used": [],
                "coverage_status": "NO_DATA",
                "missing_ranges": [{"start_date": str(as_of), "end_date": str(as_of)}],
                "next_repair_tool": repair_action["tool"],
                "repair_action": repair_action,
                "verification_action": _fundamental_verification_action(
                    symbols=symbols,
                    as_of_date=str(as_of),
                    fields=fields,
                    include_daily_basic=include_daily_basic,
                    include_financials=include_financials,
                ),
                "pit_rule": "visible_date <= as_of_date",
            },
        })

    try:
        frame = load_fundamentals_asof(
            lake,
            as_of_date=str(as_of),
            symbols=symbols or None,
            fields=fields,
            include_daily_basic=include_daily_basic,
            include_financials=include_financials,
        )
    except Exception as exc:
        return _with_query_evidence_status({
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "ERROR",
                "message": str(exc),
                "as_of_date": str(as_of),
                "requested_symbols": symbols,
                "datasets_used": datasets_used,
            },
        })

    if frame.empty:
        repair_action = _fundamental_repair_action(
            fields=fields,
            symbols=symbols,
            as_of_date=str(as_of),
            missing_symbols=symbols,
            missing_fields={field: symbols for field in fields},
        )
        return _with_query_evidence_status({
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "NO_DATA",
                "as_of_date": str(as_of),
                "requested_symbols": symbols,
                "datasets_used": datasets_used,
                "coverage_status": "NO_DATA",
                "missing_ranges": [{"start_date": str(as_of), "end_date": str(as_of)}],
                "next_repair_tool": repair_action["tool"],
                "repair_action": repair_action,
                "verification_action": _fundamental_verification_action(
                    symbols=symbols,
                    as_of_date=str(as_of),
                    fields=fields,
                    include_daily_basic=include_daily_basic,
                    include_financials=include_financials,
                ),
                "pit_rule": "visible_date <= as_of_date",
            },
        })

    total_rows = len(frame)
    output = frame.head(MAX_FUNDAMENTAL_ROWS).reset_index(drop=True)
    returned_symbols = (
        output["symbol"].dropna().astype(str).tolist()
        if "symbol" in output.columns
        else []
    )
    missing_symbols = [symbol for symbol in symbols if symbol not in set(returned_symbols)]
    missing_fields = {
        field: returned_symbols
        for field in fields
        if field not in output.columns or output[field].isna().all()
    }
    status = "OK"
    if missing_symbols or missing_fields:
        status = "PARTIAL_COVERAGE"
    partial_repair_action: dict[str, Any] | None = None
    verification_action: dict[str, Any] | None = None
    if status == "PARTIAL_COVERAGE":
        partial_repair_action = _fundamental_repair_action(
            fields=fields,
            symbols=symbols,
            as_of_date=str(as_of),
            missing_symbols=missing_symbols,
            missing_fields=missing_fields,
        )
        verification_action = _fundamental_verification_action(
            symbols=symbols,
            as_of_date=str(as_of),
            fields=fields,
            include_daily_basic=include_daily_basic,
            include_financials=include_financials,
        )
    return _with_query_evidence_status({
        "rows": records_jsonable(output),
        "metadata": {
            "point_in_time": True,
            "status": status,
            "as_of_date": str(as_of),
            "requested_symbols": symbols,
            "returned": len(output),
            "total_rows": total_rows,
            "truncated": total_rows > len(output),
            "missing_symbols": missing_symbols,
            "missing_fields": missing_fields,
            "pit_rule": (
                "financial visible_date <= as_of_date; "
                "daily_basic trade_date <= as_of_date"
            ),
            "datasets_used": datasets_used,
            "coverage_status": status,
            "missing_ranges": (
                [{"start_date": str(as_of), "end_date": str(as_of)}]
                if status == "PARTIAL_COVERAGE"
                else []
            ),
            "next_repair_tool": (
                partial_repair_action["tool"] if partial_repair_action else None
            ),
            "repair_action": partial_repair_action,
            "verification_action": verification_action,
        },
    })


def _fundamental_repair_action(
    *,
    fields: list[str],
    symbols: list[str],
    as_of_date: str,
    missing_symbols: list[str],
    missing_fields: dict[str, Any],
) -> dict[str, Any]:
    fields_to_fetch = sorted({*fields, *[str(field) for field in missing_fields]})
    source_index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())
    endpoint_fields: dict[str, tuple[FieldSourceSpec, list[str]]] = {}
    unknown_fields: list[str] = []
    event_fields: list[dict[str, str | None]] = []
    ambiguous_fields: list[dict[str, Any]] = []
    for field in fields_to_fetch:
        source = source_index.best_source_for_field(field, target_frequency=Frequency.DAILY)
        candidates = source_index.sources_for_field(field)
        if source is None:
            if candidates:
                ambiguous_fields.append(
                    {
                        "field": field,
                        "candidates": [_source_candidate_metadata(item) for item in candidates],
                    }
                )
                continue
            unknown_fields.append(field)
            continue
        if source.fill_policy is FillPolicy.NO_FILL:
            event_fields.append({"field": field, "api_name": source.api_name})
            continue
        if source.fill_policy not in {FillPolicy.EXACT, FillPolicy.ASOF_SNAPSHOT}:
            unknown_fields.append(field)
            continue
        _source, source_fields = endpoint_fields.setdefault(source.api_name, (source, []))
        source_fields.append(field)
    if ambiguous_fields and not endpoint_fields:
        first = ambiguous_fields[0]
        return {
            "type": "AMBIGUOUS_FIELD_SOURCE",
            "tool": "list_tushare_capabilities",
            "reason": "multiple_candidate_sources",
            "field": first["field"],
            "candidates": first["candidates"],
            "fields": [item["field"] for item in ambiguous_fields],
        }
    if event_fields and not endpoint_fields:
        return {
            "type": "event_transform_required",
            "tool": None,
            "reason": "event_field_requires_explicit_transform",
            "fields": [str(item["field"]) for item in event_fields],
            "candidates": event_fields,
        }
    if unknown_fields and not endpoint_fields:
        return {
            "type": "capability_discovery_required",
            "tool": "list_tushare_capabilities",
            "reason": "unknown_fundamental_field_source",
            "fields": unknown_fields,
        }

    fetch_symbols = missing_symbols or symbols
    fetch_items: list[dict[str, Any]] = []
    for _api_name, (source, endpoint_specific_fields) in sorted(endpoint_fields.items()):
        fetch_items.append(
            {
                "api_name": source.api_name,
                "symbols": fetch_symbols,
                "fields": fetch_columns_for_source(
                    source,
                    sorted(endpoint_specific_fields),
                ),
                "start_date": _fundamental_fetch_start(as_of_date, source),
                "end_date": as_of_date,
            }
        )
    if ambiguous_fields:
        return {
            "type": "AMBIGUOUS_FIELD_SOURCE",
            "tool": "list_tushare_capabilities",
            "reason": "multiple_candidate_sources",
            "fields": [item["field"] for item in ambiguous_fields],
            "candidate_fetch_items": fetch_items,
            "ambiguous_fields": ambiguous_fields,
        }
    if event_fields:
        return {
            "type": "event_transform_required",
            "tool": None,
            "reason": "event_field_requires_explicit_transform",
            "fields": [str(item["field"]) for item in event_fields],
            "candidates": event_fields,
            "candidate_fetch_items": fetch_items,
        }
    if unknown_fields:
        return {
            "type": "capability_discovery_required",
            "tool": "list_tushare_capabilities",
            "reason": "unknown_fundamental_field_source",
            "fields": unknown_fields,
            "candidate_fetch_items": fetch_items,
        }
    return {
        "type": "fetch_missing_data",
        "tool": "run_tushare_fetch",
        "reason": _fundamental_repair_reason(
            [source for source, _fields in endpoint_fields.values()]
        ),
        "fetch_items": fetch_items,
        "execute_plan": True,
    }


def _fundamental_verification_action(
    *,
    symbols: list[str],
    as_of_date: str,
    fields: list[str],
    include_daily_basic: bool,
    include_financials: bool,
) -> dict[str, Any]:
    return {
        "tool": "query_fundamentals_pit",
        "input": {
            "symbols": symbols,
            "as_of_date": as_of_date,
            "fields": fields,
            "include_daily_basic": include_daily_basic,
            "include_financials": include_financials,
        },
    }


def _fundamental_fetch_start(as_of_date: str, source: FieldSourceSpec) -> str:
    if source.fill_policy is FillPolicy.EXACT:
        return as_of_date
    parsed = pd.to_datetime(as_of_date).date()
    return f"{parsed - pd.DateOffset(years=1):%Y%m%d}"


def _fundamental_repair_reason(sources: list[FieldSourceSpec]) -> str:
    if {source.api_name for source in sources} == {"daily_basic"}:
        return "missing_daily_basic_coverage"
    if any(source.fill_policy is FillPolicy.ASOF_SNAPSHOT for source in sources):
        return "missing_financial_statement_coverage"
    return "missing_fundamental_coverage"


def _source_candidate_metadata(source: FieldSourceSpec) -> dict[str, Any]:
    return {
        "api_name": source.api_name,
        "raw_dataset_name": source.raw_dataset_name,
        "frequency": source.frequency.value,
        "fill_policy": source.fill_policy.value,
        "pit_safe": source.pit_safe,
    }


query_fundamentals_pit_tool: AgentTool = tool(
    ToolSpec(
        name="query_fundamentals_pit",
        description="按 point-in-time 语义查询财务数据。",
        permission=PermissionLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}},
                "symbol": {"type": "string"},
                "as_of_date": {"type": "string"},
                "fields": {"type": "array", "items": {"type": "string"}},
                "include_daily_basic": {"type": "boolean"},
                "include_financials": {"type": "boolean"},
            },
            "required": ["as_of_date"],
        },
        deterministic=False,
        timeout_seconds=30,
    ),
    fn=_query_fundamentals_pit,
)

# ── query_macro_series_pit ───────────────────────────────────────────────────


def _query_macro_series_pit(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return _with_query_evidence_status({
            "rows": [],
            "metadata": {
                "status": "NOT_AVAILABLE",
                "point_in_time": True,
                "message": "data lake not wired",
            },
        })
    dataset = str(input_data.get("dataset") or "").strip()
    if not dataset:
        return _with_query_evidence_status({
            "rows": [],
            "metadata": {
                "status": "INVALID_REQUEST",
                "point_in_time": True,
                "message": "dataset is required",
            },
        })
    fields_value = input_data.get("fields")
    fields = [str(field) for field in fields_value] if isinstance(fields_value, list) else None
    strict_pit = bool(input_data.get("strict_pit", True))
    frame, metadata = load_macro_series_asof(
        lake,
        dataset=dataset,
        as_of_date=str(input_data.get("as_of_date", _today_yyyymmdd())),
        start_date=input_data.get("start_date"),
        end_date=input_data.get("end_date"),
        fields=fields,
    )
    if metadata.get("status") == "INVALID_REQUEST":
        known_datasets = sorted(MACRO_DATASETS)
        repair_action: dict[str, Any] = {
            "type": "fix_request_argument",
            "tool": "list_tushare_capabilities",
            "reason": "unknown_macro_dataset",
            "invalid_dataset": dataset,
            "known_datasets": known_datasets,
            "suggested_dataset": _suggest_macro_dataset(dataset, known_datasets),
        }
        metadata = {
            **metadata,
            "coverage_status": "INVALID_REQUEST",
            "known_datasets": known_datasets,
            "next_repair_tool": "list_tushare_capabilities",
            "repair_action": repair_action,
            "suggested_repair": repair_action,
        }
    if metadata.get("status") == "NO_DATA":
        start = str(input_data.get("start_date") or input_data.get("as_of_date", _today_yyyymmdd()))
        end = str(input_data.get("end_date") or input_data.get("as_of_date", _today_yyyymmdd()))
        repair_action = {
            "type": "fetch_missing_data",
            "tool": "run_tushare_fetch",
            "reason": "known_macro_dataset_missing_local_data",
            "fetch_items": [
                {
                    "api_name": dataset,
                    "start_date": start,
                    "end_date": end,
                }
            ],
            "execute_plan": True,
        }
        metadata = {
            **metadata,
            "coverage_status": "NO_DATA",
            "missing_ranges": [{"start_date": start, "end_date": end}],
            "next_repair_tool": "run_tushare_fetch",
            "known_datasets": sorted(MACRO_DATASETS),
            "repair_action": repair_action,
            "verification_action": _macro_verification_action(
                dataset=dataset,
                as_of_date=str(input_data.get("as_of_date", _today_yyyymmdd())),
                start_date=start,
                end_date=end,
                strict_pit=strict_pit,
            ),
        }
    if not frame.empty:
        actual_start, actual_end = _macro_actual_window(frame)
        metadata = {
            **metadata,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "visible_window": {
                "actual_start": actual_start,
                "actual_end": actual_end,
                "pit_safe": metadata.get("pit_safe"),
            },
        }
    if metadata.get("pit_safe") is False and strict_pit:
        metadata = {
            **metadata,
            "status": "PIT_NOT_VALIDATED",
            "coverage_status": "PARTIAL_COVERAGE",
            "warning": (
                "This dataset uses conservative visibility approximation; do not use "
                "for production backtests unless release timing is validated."
            ),
        }
        return _with_query_evidence_status({"rows": [], "metadata": metadata})
    if metadata.get("pit_safe") is False:
        metadata = {
            **metadata,
            "warning": (
                "This dataset uses conservative visibility approximation; use for "
                "explanatory research only unless release timing is validated."
            ),
        }
    total_rows = len(frame)
    output = frame.head(MAX_MACRO_ROWS).reset_index(drop=True)
    metadata = {
        **metadata,
        "returned": len(output),
        "total_rows": total_rows,
        "truncated": total_rows > len(output),
    }
    return _with_query_evidence_status(
        {"rows": macro_records_jsonable(output), "metadata": metadata}
    )


def _suggest_macro_dataset(dataset: str, known_datasets: list[str]) -> str | None:
    normalized = dataset.lower().replace("-", "_")
    if normalized == "cpi" and "cn_cpi" in known_datasets:
        return "cn_cpi"
    if normalized == "ppi" and "cn_ppi" in known_datasets:
        return "cn_ppi"
    if normalized == "gdp" and "cn_gdp" in known_datasets:
        return "cn_gdp"
    for known in known_datasets:
        if normalized in known or known in normalized:
            return known
    return None


def _macro_verification_action(
    *,
    dataset: str,
    as_of_date: str,
    start_date: str,
    end_date: str,
    strict_pit: bool,
) -> dict[str, Any]:
    return {
        "tool": "query_macro_series_pit",
        "input": {
            "dataset": dataset,
            "as_of_date": as_of_date,
            "start_date": start_date,
            "end_date": end_date,
            "strict_pit": strict_pit,
        },
    }


def _macro_actual_window(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    if "month" in frame.columns:
        column = frame.loc[:, "month"]
        if isinstance(column, pd.DataFrame):
            column = column.iloc[:, 0]
        values = column.astype(str).dropna()
        if values.empty:
            return None, None
        return str(values.min()), str(values.max())
    if "quarter" in frame.columns:
        column = frame.loc[:, "quarter"]
        if isinstance(column, pd.DataFrame):
            column = column.iloc[:, 0]
        values = column.astype(str).dropna()
        if values.empty:
            return None, None
        return str(values.min()), str(values.max())
    if "date" in frame.columns:
        column = frame.loc[:, "date"]
        if isinstance(column, pd.DataFrame):
            column = column.iloc[:, 0]
        values = pd.to_datetime(column, errors="coerce").dropna()
        if values.empty:
            return None, None
        return str(values.min().date()), str(values.max().date())
    if "period_date" in frame.columns:
        values = pd.to_datetime(frame["period_date"], errors="coerce").dropna()
    elif "visible_date" in frame.columns:
        values = pd.to_datetime(frame["visible_date"], errors="coerce").dropna()
    else:
        return None, None
    if values.empty:
        return None, None
    return str(values.min().date()), str(values.max().date())


query_macro_series_pit_tool: AgentTool = tool(
    ToolSpec(
        name="query_macro_series_pit",
        description="按 point-in-time 语义查询结构化宏观时间序列。",
        permission=PermissionLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "as_of_date": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "fields": {"type": "array", "items": {"type": "string"}},
                "strict_pit": {"type": "boolean"},
            },
            "required": ["dataset", "as_of_date"],
        },
        deterministic=False,
        timeout_seconds=30,
    ),
    fn=_query_macro_series_pit,
)


def build_query_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    return [
        tool(
            list_data_catalog_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _list_data_catalog, input_data, context
            ),
        ),
        tool(
            query_bars_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _query_bars, input_data, context
            ),
        ),
        tool(
            query_fundamentals_pit_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _query_fundamentals_pit, input_data, context
            ),
        ),
        tool(
            query_macro_series_pit_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _query_macro_series_pit, input_data, context
            ),
        ),
    ]


def _fundamental_datasets_used(
    lake: DataLake,
    include_daily_basic: bool,
    include_financials: bool,
) -> list[str]:
    datasets: list[str] = []
    if include_daily_basic and lake.dataset_path("raw", DAILY_BASIC_DATASET).exists():
        datasets.append(DAILY_BASIC_DATASET)
    if include_financials:
        for dataset in FINANCIAL_DATASETS.values():
            if lake.dataset_path("raw", dataset).exists():
                datasets.append(dataset)
    return datasets
