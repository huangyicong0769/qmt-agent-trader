"""Data tools: list_data_catalog, query_universe, query_bars, query_fundamentals_pit.

These tools are stubs when the underlying data layer is unavailable — they
return `NOT_AVAILABLE` rather than crashing the Agent loop.
"""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.catalog import visible_dataset_names
from qmt_agent_trader.data.storage import DataLake

_lake: DataLake | None = None


def set_data_lake(lake: DataLake) -> None:
    global _lake
    _lake = lake


def _get_lake() -> DataLake | None:
    return _lake


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
    end = input_data.get("end_date", "20260624")
    fields = input_data.get(
        "fields",
        ["symbol", "trade_date", "open", "high", "low", "close", "volume"],
    )

    try:
        bars = load_daily_bars(lake, start=start, end=end, symbols=symbols or None)
        if bars.empty:
            metadata: dict[str, Any] = {"returned": 0}
            if symbols:
                metadata["requested_symbols"] = symbols
                metadata["reason"] = "no matching bars"
            else:
                metadata["status"] = "empty"
            return {"rows": [], "metadata": metadata}
        cols = [c for c in fields if c in bars.columns]
        # Limit rows for agent safety
        output = bars[cols].head(2000).to_dict(orient="records")
        return {
            "rows": output,
            "metadata": {
                "requested_symbols": symbols,
                "requested": len(symbols),
                "returned": len(output),
            },
        }
    except Exception as exc:
        return {"rows": [], "metadata": {"error": str(exc)}}


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
