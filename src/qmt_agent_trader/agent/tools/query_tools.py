"""Data tools: list_data_catalog, query_universe, query_bars, query_fundamentals_pit.

These tools are stubs when the underlying data layer is unavailable — they
return `NOT_AVAILABLE` rather than crashing the Agent loop.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import date, datetime
from typing import Any

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import SHANGHAI_TZ
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.catalog import visible_dataset_names
from qmt_agent_trader.data.storage import DataLake

_lake: DataLake | None = None
_lake_var: ContextVar[DataLake | None] = ContextVar("query_tool_lake", default=None)


def set_data_lake(lake: DataLake) -> None:
    global _lake
    _lake = lake


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
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
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


def _query_universe(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_AVAILABLE", "message": "data lake not wired"}

    as_of = input_data.get("as_of_date", "20200101")
    universe_type = input_data.get("universe_type", "stock")
    filters = input_data.get("filters", {})
    exclude_st = filters.get("exclude_st", True)
    exclude_suspended = filters.get("exclude_suspended", True)
    filters.get("min_listed_days", 60)

    try:
        bars = load_daily_bars(lake, end=as_of)
        if bars.empty:
            return {"status": "NOT_AVAILABLE", "symbols": [], "metadata": {"reason": "no data"}}

        recent = bars[bars["trade_date"] == bars["trade_date"].max()]
        symbols = recent["symbol"].astype(str).tolist()
        if exclude_st:
            st_mask = recent.set_index("symbol")["st"]
            symbols = [s for s in symbols if not st_mask.get(s, False)]
        if exclude_suspended:
            susp_mask = recent.set_index("symbol")["suspended"]
            symbols = [s for s in symbols if not susp_mask.get(s, False)]

        return {
            "symbols": symbols[:2000],
            "metadata": {
                "as_of_date": str(recent["trade_date"].iloc[0]),
                "universe_type": universe_type,
                "count": len(symbols),
            },
        }
    except Exception as exc:
        return {"status": "NOT_AVAILABLE", "symbols": [], "metadata": {"error": str(exc)}}


query_universe_tool: AgentTool = tool(
    ToolSpec(
        name="query_universe",
        description="查询某日可投资股票池 / ETF 池。",
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_query_universe,
)

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

    try:
        bars = load_daily_bars(lake, start=start, end=end, symbols=symbols or None)
        if bars.empty:
            coverage_metadata = _bars_coverage_metadata(symbols, bars, end)
            metadata: dict[str, Any] = {
                "requested_start_date": str(start),
                "requested_end_date": str(end),
                "returned": 0,
                **coverage_metadata,
            }
            if symbols:
                metadata["requested_symbols"] = symbols
                metadata["reason"] = "no matching bars"
            else:
                metadata["status"] = "NO_MATCHING_BARS"
            return {"rows": [], "metadata": metadata}
        cols = [c for c in fields if c in bars.columns]
        # Limit rows for agent safety
        output = bars[cols].head(2000).to_dict(orient="records")
        coverage_metadata = _bars_coverage_metadata(symbols, bars, end)
        return {
            "rows": output,
            "metadata": {
                "requested_symbols": symbols,
                "requested": len(symbols),
                "requested_start_date": str(start),
                "requested_end_date": str(end),
                "actual_start_date": str(bars["trade_date"].min()),
                "actual_end_date": str(bars["trade_date"].max()),
                "data_freshness": _freshness(str(bars["trade_date"].max()), str(end)),
                "returned": len(output),
                "total_rows": len(bars),
                "identity_fields_forced": True,
                **coverage_metadata,
            },
        }
    except Exception as exc:
        return {"rows": [], "metadata": {"error": str(exc)}}


def _bar_output_fields(requested_fields: Any) -> list[str]:
    raw_fields = requested_fields if isinstance(requested_fields, list) else []
    fields: list[str] = []
    for field in ["symbol", "trade_date", *[str(field) for field in raw_fields]]:
        if field not in fields:
            fields.append(field)
    return fields


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
            "actual_start_date": str(actual_start),
            "actual_end_date": str(actual_end),
            "data_freshness": data_freshness,
        }

    if len(missing_symbols) == len(symbols):
        status = "NO_MATCHING_BARS"
    elif missing_symbols or stale_symbols:
        status = "PARTIAL_COVERAGE"
    else:
        status = "OK"
    return {
        "status": status,
        "coverage_by_symbol": coverage_by_symbol,
        "missing_symbols": missing_symbols,
        "stale_symbols": stale_symbols,
        "covered_symbols": covered_symbols,
    }


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
    return {
        "rows": [],
        "metadata": {
            "point_in_time": True,
            "status": "NOT_IMPLEMENTED",
            "message": "PIT fundamentals data is not yet available; use daily bars for now.",
        },
    }


query_fundamentals_pit_tool: AgentTool = tool(
    ToolSpec(
        name="query_fundamentals_pit",
        description="按 point-in-time 语义查询财务数据。",
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
        timeout_seconds=30,
    ),
    fn=_query_fundamentals_pit,
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
            query_universe_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _query_universe, input_data, context
            ),
        ),
        tool(
            query_bars_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _query_bars, input_data, context
            ),
        ),
    ]
