"""Profile research data/tool paths on the local data lake.

Default quick mode keeps inputs bounded and does not require a Tushare token.
Use --full for heavier all-market probes.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.query_tools import query_bars_tool, set_data_lake
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.service import evaluate_factor
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.registry import StrategyRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run bounded quick probes.")
    parser.add_argument("--full", action="store_true", help="Run heavier all-market probes.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    full = bool(args.full and not args.quick)

    settings = get_settings()
    lake = DataLake(
        root=settings.resolved_data_dir / "lake",
        duckdb_path=settings.resolved_data_dir / "qmt_agent_trader.duckdb",
    )
    report_root = Path("reports/perf")
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if not _has_bars(lake):
        payload = {
            "status": "NO_LOCAL_DATA",
            "message": "raw tushare_daily/tushare_fund_daily parquet is not available",
            "lake_root": str(lake.root),
        }
        _write_reports(report_root, stamp, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    bounds = _date_bounds(lake)
    end = args.end or bounds["end"]
    start = args.start or _quick_start(bounds["start"], end, full=full)
    symbols = args.symbols or _sample_symbols(lake, start=start, end=end, limit=20 if full else 5)
    set_data_lake(lake)

    results: list[dict[str, Any]] = []
    results.append(
        _measure(
            "load_daily_bars_small_window",
            lambda: load_daily_bars(lake, start=start, end=end, symbols=symbols),
        )
    )
    if full:
        results.append(
            _measure(
                "load_daily_bars_all_market",
                lambda: load_daily_bars(lake, start=start, end=end, symbols=None),
            )
        )
    results.append(
        _measure(
            "query_bars_all_market_limit_2000",
            lambda: query_bars_tool.run(
                {
                    "start_date": start,
                    "end_date": end,
                    "limit": 2000,
                    "include_trade_state": False,
                },
                ToolContext(run_id="profile-query-bars", requested_by_llm=False),
            ),
        )
    )
    results.append(
        _measure(
            "evaluate_factor_candidate_builtin",
            lambda: evaluate_factor(
                lake,
                name="momentum_20d",
                start=start,
                end=end,
                symbols=symbols,
                window_days=20,
                step_days=20,
                quantile=0.5,
            ).validation.as_dict(),
        )
    )
    results.append(
        _measure(
            "run_backtest_builtin",
            lambda: run_strategy_backtest(
                lake,
                StrategyRegistry(settings.resolved_data_dir / "strategies"),
                StrategyBacktestConfig(
                    strategy_id="profile_momentum_20d",
                    factor_name="momentum_20d",
                    start_date=start,
                    end_date=end,
                    symbols=symbols,
                    top_n=min(5, max(1, len(symbols))),
                ),
                reports_dir=Path("reports/research"),
            ).model_dump(mode="json"),
        )
    )
    payload = {
        "status": "OK",
        "mode": "full" if full else "quick",
        "start": start,
        "end": end,
        "symbols": symbols,
        "results": results,
    }
    _write_reports(report_root, stamp, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _measure(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:
        return {
            "name": name,
            "status": "ERROR",
            "duration_seconds": round(time.perf_counter() - started, 4),
            "error": str(exc),
        }
    duration = round(time.perf_counter() - started, 4)
    return {
        "name": name,
        "status": "OK",
        "duration_seconds": duration,
        **_shape(value),
    }


def _shape(value: Any) -> dict[str, Any]:
    if hasattr(value, "empty") and hasattr(value, "columns"):
        return {
            "rows": len(value),
            "columns": list(value.columns),
            "symbols": int(value["symbol"].nunique()) if "symbol" in value.columns else None,
            "dates": int(value["trade_date"].nunique()) if "trade_date" in value.columns else None,
        }
    if isinstance(value, dict):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        return {
            "rows": len(value.get("rows", [])) if isinstance(value.get("rows"), list) else None,
            "cache_hit": value.get("cache_hit"),
            "timeout_seconds_used": value.get("timeout_seconds_used"),
            "metadata": metadata,
        }
    return {"value_type": type(value).__name__}


def _has_bars(lake: DataLake) -> bool:
    return any(
        lake.dataset_path("raw", name).exists()
        for name in ("tushare_daily", "tushare_fund_daily")
    )


def _date_bounds(lake: DataLake) -> dict[str, str]:
    frames: list[Any] = []
    for name in ("tushare_daily", "tushare_fund_daily"):
        path = lake.dataset_path("raw", name)
        if not path.exists():
            continue
        escaped = str(path).replace("'", "''")
        frames.append(
            lake.query_external(
                f"""
                SELECT
                    min({_date_key_sql("trade_date")}) AS start_date,
                    max({_date_key_sql("trade_date")}) AS end_date
                FROM read_parquet('{escaped}')
                """
            )
        )
    starts = [str(frame.iloc[0]["start_date"]) for frame in frames if not frame.empty]
    ends = [str(frame.iloc[0]["end_date"]) for frame in frames if not frame.empty]
    return {"start": min(starts), "end": max(ends)}


def _sample_symbols(lake: DataLake, *, start: str, end: str, limit: int) -> list[str]:
    for name in ("tushare_daily", "tushare_fund_daily"):
        path = lake.dataset_path("raw", name)
        if not path.exists():
            continue
        escaped = str(path).replace("'", "''")
        frame = lake.query_external(
            f"""
            SELECT DISTINCT ts_code
            FROM read_parquet('{escaped}')
            WHERE {_date_key_sql("trade_date")} >= $start_date
              AND {_date_key_sql("trade_date")} <= $end_date
            ORDER BY ts_code
            LIMIT {int(limit)}
            """,
            {"start_date": start, "end_date": end},
        )
        symbols = [str(item) for item in frame["ts_code"].tolist()]
        if symbols:
            return symbols
    return []


def _quick_start(first: str, end: str, *, full: bool) -> str:
    if full:
        return first
    end_time = time.strptime(end, "%Y%m%d")
    end_seconds = time.mktime(end_time)
    start_seconds = max(time.mktime(time.strptime(first, "%Y%m%d")), end_seconds - 120 * 86400)
    return time.strftime("%Y%m%d", time.localtime(start_seconds))


def _date_key_sql(column: str) -> str:
    return (
        "COALESCE("
        f"strftime(try_strptime(CAST({column} AS VARCHAR), '%Y%m%d'), '%Y%m%d'), "
        f"strftime(TRY_CAST({column} AS DATE), '%Y%m%d'), "
        f"substr(regexp_replace(CAST({column} AS VARCHAR), '[^0-9]', '', 'g'), 1, 8)"
        ")"
    )


def _write_reports(root: Path, stamp: str, payload: dict[str, Any]) -> None:
    json_path = root / f"research_tools_{stamp}.json"
    md_path = root / f"research_tools_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    lines = [f"# Research Tool Profile {stamp}", "", f"Status: {payload.get('status')}", ""]
    for item in payload.get("results", []):
        lines.append(
            f"- {item.get('name')}: status={item.get('status')} "
            f"duration_seconds={item.get('duration_seconds')}"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
