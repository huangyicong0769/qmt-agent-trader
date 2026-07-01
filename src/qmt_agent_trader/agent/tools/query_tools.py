"""Data tools: list_data_catalog, query_universe, query_bars, query_fundamentals_pit.

These tools are stubs when the underlying data layer is unavailable — they
return `NOT_AVAILABLE` rather than crashing the Agent loop.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextvars import ContextVar
from datetime import date, datetime
from typing import Any

import pandas as pd

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import SHANGHAI_TZ
from qmt_agent_trader.data.bars import (
    enrich_trade_states,
    normalize_tushare_daily,
)
from qmt_agent_trader.data.catalog import visible_dataset_names
from qmt_agent_trader.data.fundamentals import (
    DAILY_BASIC_DATASET,
    DEFAULT_FUNDAMENTAL_FIELDS,
    FINANCIAL_DATASETS,
    load_fundamentals_asof,
    records_jsonable,
)
from qmt_agent_trader.data.macro import MACRO_DATASETS
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

    as_of = input_data.get("as_of_date", _today_yyyymmdd())
    universe_type = input_data.get("universe_type", "stock")
    filters = input_data.get("filters", {})
    theme = str(filters.get("theme", "")).lower()
    exclude_st = filters.get("exclude_st", True)
    exclude_suspended = filters.get("exclude_suspended", True)
    min_listed_days = int(filters.get("min_listed_days", 60))

    try:
        if theme == "cyclical":
            return build_theme_universe(
                lake,
                as_of=as_of,
                theme=theme,
                exclude_st=bool(exclude_st),
                exclude_suspended=bool(exclude_suspended),
                min_listed_days=min_listed_days,
            )
        bars = _load_recent_bars_for_universe(lake, end=str(as_of))
        if bars.empty:
            return {"status": "NOT_AVAILABLE", "symbols": [], "metadata": {"reason": "no data"}}

        recent = bars
        symbols = recent["symbol"].astype(str).tolist()
        if exclude_st:
            st_mask = recent.set_index("symbol")["st"]
            symbols = [s for s in symbols if not st_mask.get(s, False)]
        if exclude_suspended:
            susp_mask = recent.set_index("symbol")["suspended"]
            symbols = [s for s in symbols if not susp_mask.get(s, False)]

        return {
            "status": "OK",
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
        description=(
            "查询某日可投资股票池 / ETF 池。支持 filters.theme='cyclical' "
            "基于 tushare_stock_basic 行业/名称映射构造可复现顺周期篮子。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "as_of_date": {"type": "string"},
                "universe_type": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "theme": {
                            "type": "string",
                            "description": "Use 'cyclical' for the reproducible cyclical basket.",
                        },
                        "exclude_st": {"type": "boolean"},
                        "exclude_suspended": {"type": "boolean"},
                        "min_listed_days": {"type": "integer"},
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": False,
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_query_universe,
)


CYCLICAL_INDUSTRIES = {
    "银行",
    "证券",
    "保险",
    "多元金融",
    "房地产",
    "钢铁",
    "煤炭",
    "有色金属",
    "工业金属",
    "贵金属",
    "基础化工",
    "化工",
    "建筑材料",
    "建材",
    "建筑装饰",
    "机械设备",
    "汽车",
    "石油石化",
    "交通运输",
    "航运港口",
}


def _load_recent_bars_for_universe(lake: DataLake, *, end: str) -> Any:
    frames: list[Any] = []
    end_key = _date_key(end)
    for dataset in ("tushare_daily", "tushare_fund_daily"):
        path = lake.dataset_path("raw", dataset)
        if not path.exists():
            continue
        escaped_path = str(path).replace("'", "''")
        frame = lake.query_parquet(
            f"""
            WITH source AS (
                SELECT *
                FROM read_parquet('{escaped_path}')
                WHERE CAST(trade_date AS VARCHAR) <= $end_date
            ),
            latest AS (
                SELECT max(CAST(trade_date AS VARCHAR)) AS trade_date
                FROM source
            )
            SELECT source.*
            FROM source, latest
            WHERE CAST(source.trade_date AS VARCHAR) = latest.trade_date
            """,
            {"end_date": end_key},
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return normalize_tushare_daily(pd.DataFrame())
    recent = normalize_tushare_daily(pd.concat(frames, ignore_index=True))
    return _apply_fast_universe_state(lake, recent)


def build_theme_universe(
    lake: DataLake,
    *,
    as_of: str,
    theme: str,
    exclude_st: bool = True,
    exclude_suspended: bool = True,
    min_listed_days: int = 60,
) -> dict[str, Any]:
    recent = _load_recent_bars_for_universe(lake, end=as_of)
    if recent.empty:
        return {"status": "NOT_AVAILABLE", "symbols": [], "metadata": {"reason": "no data"}}
    if theme == "cyclical":
        return _query_theme_universe(
            lake,
            recent,
            as_of=as_of,
            theme=theme,
            exclude_st=exclude_st,
            exclude_suspended=exclude_suspended,
            min_listed_days=min_listed_days,
        )
    return {
        "status": "INVALID_REQUEST",
        "symbols": [],
        "metadata": {"reason": "unsupported_theme", "theme": theme},
    }


def _apply_fast_universe_state(lake: DataLake, recent: Any) -> Any:
    if recent.empty or "symbol" not in recent.columns:
        return recent
    stock_basic_path = lake.dataset_path("raw", "tushare_stock_basic")
    if not stock_basic_path.exists():
        return recent
    stock_basic = lake.read_parquet("raw", "tushare_stock_basic")
    if stock_basic.empty or not {"ts_code", "name"}.issubset(stock_basic.columns):
        return recent
    st_symbols = set(
        stock_basic.loc[
            stock_basic["name"].astype(str).str.contains("ST", case=False, na=False),
            "ts_code",
        ].astype(str)
    )
    if not st_symbols:
        return recent
    enriched = recent.copy()
    enriched["st"] = enriched["st"] | enriched["symbol"].astype(str).isin(st_symbols)
    return enriched


def _query_theme_universe(
    lake: DataLake,
    recent: Any,
    *,
    as_of: str,
    theme: str,
    exclude_st: bool,
    exclude_suspended: bool,
    min_listed_days: int,
) -> dict[str, Any]:
    if not lake.dataset_path("raw", "tushare_stock_basic").exists():
        return {
            "status": "BLOCKED",
            "symbols": [],
            "metadata": {
                "theme": theme,
                "reason": "missing_stock_basic",
                "next_repair_tool": "run_remote_data_update",
            },
        }
    stock_basic = lake.read_parquet("raw", "tushare_stock_basic")
    if stock_basic.empty or "ts_code" not in stock_basic.columns:
        return {
            "status": "BLOCKED",
            "symbols": [],
            "metadata": {
                "theme": theme,
                "reason": "invalid_stock_basic",
                "next_repair_tool": "run_remote_data_update",
            },
        }

    recent_by_symbol = recent.set_index("symbol")
    as_of_date = _parse_date(as_of)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for row in stock_basic.to_dict(orient="records"):
        symbol = str(row.get("ts_code", ""))
        name = str(row.get("name", ""))
        industry = str(row.get("industry", ""))
        reason = _theme_exclusion_reason(
            symbol=symbol,
            name=name,
            industry=industry,
            row=row,
            recent_by_symbol=recent_by_symbol,
            as_of_date=as_of_date,
            exclude_st=exclude_st,
            exclude_suspended=exclude_suspended,
            min_listed_days=min_listed_days,
        )
        if reason is not None:
            excluded.append({"symbol": symbol, "reason": reason, "industry": industry})
            continue
        selected.append({"symbol": symbol, "name": name, "industry": industry})

    selected = sorted(selected, key=lambda item: item["symbol"])
    industries = Counter(item["industry"] for item in selected)
    return {
        "status": "OK",
        "symbols": [item["symbol"] for item in selected][:2000],
        "metadata": {
            "theme": theme,
            "as_of_date": str(recent["trade_date"].iloc[0]),
            "count": len(selected),
            "industry_distribution": dict(sorted(industries.items())),
            "selection_rules": {
                "industry_source": "tushare_stock_basic",
                "included_industries": sorted(CYCLICAL_INDUSTRIES),
                "exclude_st": exclude_st,
                "exclude_suspended": exclude_suspended,
                "min_listed_days": min_listed_days,
            },
            "excluded_symbols": excluded[:2000],
        },
    }


def _theme_exclusion_reason(
    *,
    symbol: str,
    name: str,
    industry: str,
    row: dict[str, Any],
    recent_by_symbol: Any,
    as_of_date: date,
    exclude_st: bool,
    exclude_suspended: bool,
    min_listed_days: int,
) -> str | None:
    if not symbol:
        return "missing_symbol"
    if str(row.get("list_status", "L")) not in {"L", "上市"}:
        return "not_listed"
    if industry not in CYCLICAL_INDUSTRIES:
        return "industry_not_in_theme"
    if symbol not in recent_by_symbol.index:
        return "no_bar_coverage"
    recent = recent_by_symbol.loc[symbol]
    if exclude_st and (bool(recent.get("st", False)) or "ST" in name.upper()):
        return "st"
    if exclude_suspended and bool(recent.get("suspended", False)):
        return "suspended"
    list_date_raw = row.get("list_date")
    if list_date_raw:
        listed_days = (as_of_date - _parse_date(str(list_date_raw))).days
        if listed_days < min_listed_days:
            return "listed_days_below_minimum"
    return None


def _date_key(value: str) -> str:
    return _parse_date(value).strftime("%Y%m%d")

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
        return {
            "rows": [],
            "metadata": {
                "status": "INVALID_REQUEST",
                "message": f"limit must be between 1 and {MAX_BAR_LIMIT}",
            },
        }
    include_trade_state = bool(input_data.get("include_trade_state", True))
    if "enrich" in input_data:
        include_trade_state = bool(input_data["enrich"])
    order = str(input_data.get("order", "asc")).lower()
    if order not in {"asc", "desc"}:
        return {
            "rows": [],
            "metadata": {"status": "INVALID_REQUEST", "message": "order must be asc or desc"},
        }

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
        if bars.empty:
            coverage_metadata = _bars_coverage_metadata(symbols, bars, end)
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
            return {"rows": [], "metadata": metadata}
        cols = [c for c in fields if c in bars.columns]
        output = bars[cols].head(limit).to_dict(orient="records")
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
                "limit": limit,
                "backend_limited": True,
                "include_trade_state": include_trade_state,
                "read_strategy": "predicate_pushdown",
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
            "tushare_suspend",
            columns=["ts_code", "trade_date", "suspend_type"],
            start=returned_start,
            end=returned_end,
            symbols=returned_symbols,
        ),
        stk_limit=lake.read_parquet_filtered(
            "raw",
            "tushare_stk_limit",
            columns=["ts_code", "trade_date", "up_limit", "down_limit"],
            start=returned_start,
            end=returned_end,
            symbols=returned_symbols,
        ),
        namechange=lake.read_parquet_filtered(
            "raw",
            "tushare_namechange",
            columns=["ts_code", "name", "start_date", "end_date"],
            symbols=returned_symbols,
        ),
        stock_basic=lake.read_parquet_filtered(
            "raw",
            "tushare_stock_basic",
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
    for dataset in ("tushare_daily", "tushare_fund_daily"):
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
        frame = lake.query_parquet(sql, params)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _available_bar_columns(lake: DataLake, escaped_path: str) -> set[str]:
    schema = lake.query_parquet(f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')")
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
    lake = _get_lake()
    if lake is None:
        return {
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "NOT_AVAILABLE",
                "message": "data lake not wired",
            },
        }

    as_of = input_data.get("as_of_date")
    if not as_of:
        return {
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "INVALID_REQUEST",
                "message": "as_of_date is required",
            },
        }
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
        return {
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "NO_DATA",
                "as_of_date": str(as_of),
                "requested_symbols": symbols,
                "datasets_used": [],
                "coverage_status": "NO_DATA",
                "missing_ranges": [{"start_date": str(as_of), "end_date": str(as_of)}],
                "next_repair_tool": "run_fundamental_data_update",
                "pit_rule": "visible_date <= as_of_date",
            },
        }

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
        return {
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "ERROR",
                "message": str(exc),
                "as_of_date": str(as_of),
                "requested_symbols": symbols,
                "datasets_used": datasets_used,
            },
        }

    if frame.empty:
        return {
            "rows": [],
            "metadata": {
                "point_in_time": True,
                "status": "NO_DATA",
                "as_of_date": str(as_of),
                "requested_symbols": symbols,
                "datasets_used": datasets_used,
                "coverage_status": "NO_DATA",
                "missing_ranges": [{"start_date": str(as_of), "end_date": str(as_of)}],
                "next_repair_tool": "run_fundamental_data_update",
                "pit_rule": "visible_date <= as_of_date",
            },
        }

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
    return {
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
                "run_fundamental_data_update" if status == "PARTIAL_COVERAGE" else None
            ),
        },
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
        return {
            "rows": [],
            "metadata": {
                "status": "NOT_AVAILABLE",
                "point_in_time": True,
                "message": "data lake not wired",
            },
        }
    dataset = str(input_data.get("dataset") or "").strip()
    if not dataset:
        return {
            "rows": [],
            "metadata": {
                "status": "INVALID_REQUEST",
                "point_in_time": True,
                "message": "dataset is required",
            },
        }
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
        metadata = {
            **metadata,
            "known_datasets": sorted(MACRO_DATASETS),
            "next_repair_tool": "run_macro_data_update",
        }
    if metadata.get("status") == "NO_DATA":
        start = str(input_data.get("start_date") or input_data.get("as_of_date", _today_yyyymmdd()))
        end = str(input_data.get("end_date") or input_data.get("as_of_date", _today_yyyymmdd()))
        metadata = {
            **metadata,
            "coverage_status": "NO_DATA",
            "missing_ranges": [{"start_date": start, "end_date": end}],
            "next_repair_tool": "run_macro_data_update",
            "known_datasets": sorted(MACRO_DATASETS),
        }
    if metadata.get("pit_safe") is False and strict_pit:
        metadata = {
            **metadata,
            "status": "PIT_NOT_VALIDATED",
            "warning": (
                "This dataset uses conservative visibility approximation; do not use "
                "for production backtests unless release timing is validated."
            ),
        }
        return {"rows": [], "metadata": metadata}
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
    return {"rows": macro_records_jsonable(output), "metadata": metadata}


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
