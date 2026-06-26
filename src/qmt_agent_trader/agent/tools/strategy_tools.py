"""Strategy tools: create_strategy_spec, generate_strategy_code, run_backtest,
and report tool: generate_research_report."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import StrategySpec, ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import SHANGHAI_TZ, new_id, shanghai_now_iso
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.registry import FactorRegistry

_sandbox: CodeSandbox | None = None
_store: ExperimentStore | None = None
_lake: DataLake | None = None


def wire(sandbox: CodeSandbox, store: ExperimentStore, lake: DataLake) -> None:
    global _sandbox, _store, _lake
    _sandbox = sandbox
    _store = store
    _lake = lake


# ── create_strategy_spec ────────────────────────────────────────────────────


def _create_strategy_spec(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    strategy_idea = input_data.get("strategy_idea", "")
    selected_factors = input_data.get("selected_factors", [])
    universe = input_data.get("universe", "stock_etf")
    rebalance_freq = input_data.get("rebalance_frequency", "daily")
    constraints = input_data.get("constraints", {})

    strategy_id = new_id("strat")
    spec = StrategySpec(
        strategy_id=strategy_id,
        name=strategy_idea[:60] or "candidate_strategy",
        version="0.1.0",
        universe=universe,
        factors=selected_factors,
        portfolio_construction={
            "method": "equal_weight",
            "top_n": 20,
        },
        rebalance={"frequency": rebalance_freq},
        risk_constraints=constraints,
        execution_assumptions={
            "timing": "next_open",
            "slippage_model": "fixed_5bps",
        },
    )
    return {"strategy_spec": spec.model_dump(mode="json")}


create_strategy_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_strategy_spec",
        description="将策略想法和候选因子组合转成结构化 strategy spec。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_create_strategy_spec,
)

# ── generate_strategy_code ──────────────────────────────────────────────────


def _generate_strategy_code(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("strategy_spec", {})
    name = spec_data.get("name", "candidate")
    strategy_id = spec_data.get("strategy_id", new_id("strat"))
    factors = spec_data.get("factors", [])
    top_n = (
        
            spec_data.get("portfolio_construction", {}).get("top_n", 20)
            if "portfolio_construction" in spec_data
            else 20
        
    )
    warnings: list[str] = []

    if not factors:
        warnings.append("no factors selected; strategy will not generate signals")

    strategy_code = _render_strategy_code(name, factors, top_n)
    test_code = _render_strategy_test_code(name)

    sb = _sandbox
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}

    try:
        code_path = sb.write_candidate_file(
            f"strategies/drafts/{strategy_id}/strategy.py",
            strategy_code,
        )
        tests_path = sb.write_candidate_file(
            f"strategies/drafts/{strategy_id}/test_strategy.py",
            test_code,
        )
        return {
            "code_path": str(code_path),
            "tests_path": str(tests_path),
            "status": "generated",
            "warnings": warnings,
        }
    except Exception as exc:
        return {"status": "error", "warnings": [str(exc)]}


generate_strategy_code_tool: AgentTool = tool(
    ToolSpec(
        name="generate_strategy_code",
        description="根据 strategy spec 生成候选策略代码和测试。",
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_strategy_code,
)

# ── run_backtest ────────────────────────────────────────────────────────────


def _run_backtest(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _lake
    if lake is None:
        return {"status": "NOT_IMPLEMENTED", "message": "data lake not wired"}

    strategy_id = input_data.get("strategy_id", "")
    factor_name = input_data.get("factor_name", "")
    start_date = input_data.get("start_date", "20200101")
    end_date = input_data.get("end_date", _today_yyyymmdd())
    initial_cash = float(input_data.get("initial_cash", 1_000_000))
    top_n = int(input_data.get("top_n", 20))
    symbols = _requested_symbols(input_data)

    # Resolve factor_name from strategy_id if not provided
    if not factor_name and strategy_id:
        factor_name = _map_strategy_factor(strategy_id)
    if not factor_name:
        return {
            "status": "error",
            "message": "必须提供 strategy_id 或 factor_name 才能回测策略。"
        }

    # Use the research runner for a baseline backtest
    from qmt_agent_trader.backtest.research_runner import (
        FactorRankResearchConfig,
        FactorRankResearchRunner,
    )
    from qmt_agent_trader.backtest.sensitivity import SensitivityScenario
    from qmt_agent_trader.data.bars import load_daily_bars

    registry_root = _factor_registry_root(lake)
    factor_registry = FactorRegistry(registry_root)
    saved = factor_registry.get_factor(factor_name)
    if saved is None:
        return {
            "status": "FACTOR_NOT_SAVED",
            "message": f"factor '{factor_name}' is a draft or unknown; save_factor first",
        }
    factor_name = saved.factor_id

    bars = load_daily_bars(lake, symbols=symbols or None)
    if bars.empty:
        return {"status": "error", "message": "data lake is empty; run data update first"}

    # Filter bars to date range
    start_bound = pd.to_datetime(start_date).date()
    end_bound = pd.to_datetime(end_date).date()
    bars = bars[
        (bars["trade_date"] >= start_bound)
        & (bars["trade_date"] <= end_bound)
    ]
    if bars.empty:
        return {
            "status": "error",
            "message": f"no bars in range {start_date}–{end_date}",
        }

    config = FactorRankResearchConfig(
        factor_name=factor_name,
        factor_registry_root=registry_root,
        top_n=top_n,
        initial_cash=initial_cash,
    )
    runner = FactorRankResearchRunner(bars, config)
    baseline = SensitivityScenario(
        cost_multiplier=1.0,
        slippage_bps=0.0,
        execution_delay_days=1,
        top_n=top_n,
        max_single_position_pct=0.10,
    )

    result = runner.run(baseline)
    result_dict = result.as_dict()

    # Persist report
    reports_dir = Path("reports/research")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": new_id("research"),
        "created_at": shanghai_now_iso(),
        "artifact_type": "strategy_backtest",
        "title": f"Strategy backtest: {factor_name} (top_n={top_n})",
        "research_only": True,
        "approval_status": "NOT_REQUESTED",
        "live_trading_allowed": False,
        "metadata": {
            "factor_name": factor_name,
            "symbols": symbols,
            "top_n": top_n,
            "initial_cash": initial_cash,
            "start_date": start_date,
            "end_date": end_date,
        },
        "summary": {
            "baseline_total_return": result.metrics.total_return,
            "sharpe": result.metrics.sharpe,
            "max_drawdown": result.metrics.max_drawdown,
            "turnover": result.metrics.turnover,
            "diagnostic_pass": result.metrics.diagnostic_pass,
            "trade_count": len(result.trades),
            "rejected_orders": result.rejected_orders,
        },
        "performance_report": {
            "total_return": result.metrics.total_return,
            "sharpe": result.metrics.sharpe,
            "max_drawdown": result.metrics.max_drawdown,
            "turnover": result.metrics.turnover,
            "trade_count": len(result.trades),
            "fills": len(result.trades),
        },
        "trade_blotter": [
            {
                "symbol": t.symbol,
                "execution_date": t.trade_date,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
            }
            for t in result.trades
        ],
        "payload": result_dict,
    }
    report_path = reports_dir / f"{report['run_id']}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "run_id": report["run_id"],
        "status": "completed",
        "factor_name": factor_name,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "metrics": {
            "total_return": round(result.metrics.total_return, 4),
            "sharpe": round(result.metrics.sharpe, 4),
            "max_drawdown": round(result.metrics.max_drawdown, 4),
            "turnover": round(result.metrics.turnover, 4),
            "trade_count": len(result.trades),
        },
        "report_path": str(report_path),
        "cache_hit": False,
    }


run_backtest_tool: AgentTool = tool(
    ToolSpec(
        name="run_backtest",
        description=(
            "运行因子排名策略的基线回测。返回 total_return, sharpe, max_drawdown,"
            " turnover, trade_count。需提供 factor_name 或 strategy_id。"
            " 内置因子: momentum_20d, momentum_60d, reversal_5d, volatility_20d,"
            " turnover_20d, amount_zscore_20d"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "factor_name": {"type": "string"},
                "strategy_id": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "symbol": {"type": "string"},
                "code": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "initial_cash": {"type": "number"},
                "top_n": {"type": "integer"},
            },
            "required": ["factor_name"],
        },
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=False,
        timeout_seconds=120,
    ),
    fn=_run_backtest,
)

# ── generate_research_report ────────────────────────────────────────────────


def _generate_research_report(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    exp_id = input_data.get("experiment_id") or context.experiment_id or "unknown"
    run_ids = input_data.get("run_ids", [])
    sections = input_data.get("include_sections", ["summary", "metrics"])

    sb = _sandbox
    reports_root = sb.generated_root / "reports" if sb else Path("reports/research")
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"{exp_id}.md"

    store = _store
    experiment_meta = {}
    if store:
        try:
            exp = store.get_experiment(exp_id)
            experiment_meta = {
                "kind": exp.kind,
                "status": exp.status.value,
                "lessons": exp.lessons,
            }
        except Exception:
            pass

    lines = [
        f"# Research Report: {exp_id}",
        f"Generated: {shanghai_now_iso()}",
        "",
        "## Experiment",
        f"- Kind: {experiment_meta.get('kind', 'unknown')}",
        f"- Status: {experiment_meta.get('status', 'unknown')}",
        "",
    ]

    if "summary" in sections:
        lines.extend(["## Summary", "", f"Run IDs: {', '.join(run_ids)}", ""])
    if "metrics" in sections:
        lines.extend(["## Metrics", "", "(see attached backtest reports for details)", ""])
    if "lessons" in sections and experiment_meta.get("lessons"):
        lines.extend(["## Lessons Learned", ""])
        for lesson in experiment_meta["lessons"]:
            lines.append(f"- {lesson}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report_path": str(report_path), "summary": lines[0]}


generate_research_report_tool: AgentTool = tool(
    ToolSpec(
        name="generate_research_report",
        description="生成因子或策略研究报告。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_research_report,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _render_strategy_code(name: str, factors: list[str], top_n: int) -> str:
    name.replace(" ", "_").replace("-", "_").lower()
    return f'''"""Candidate strategy: {name}.
Auto-generated by Agent. REVIEW_REQUIRED before promotion.
"""

import pandas as pd


FACTORS = {factors!r}
TOP_N = {top_n}


def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate buy signals based on factor rankings."""
    if data.empty:
        return pd.DataFrame(columns=["symbol", "target_weight"])
    # Placeholder: in a full implementation, combine multiple factors into a score.
    ranked = data.head(TOP_N).copy()
    ranked["target_weight"] = 1 / max(len(ranked), 1)
    return ranked[["symbol", "target_weight"]]
'''


def _render_strategy_test_code(name: str) -> str:
    return f'''"""Tests for candidate strategy: {name}."""

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("strategy.py")
SPEC = importlib.util.spec_from_file_location("candidate_strategy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
generate_signals = MODULE.generate_signals


def test_empty_data():
    result = generate_signals(pd.DataFrame())
    assert result.empty


def test_generates_signals():
    data = pd.DataFrame({{
        "symbol": ["A", "B", "C"],
        "close": [10, 20, 30],
        "score": [0.9, 0.5, 0.1],
    }})
    result = generate_signals(data)
    assert "symbol" in result.columns
    assert "target_weight" in result.columns
'''


# ── Strategy → factor mapping (for run_backtest) ────────────────────────────


_STRATEGY_FACTORS: dict[str, str] = {
    "momentum": "momentum_20d",
    "momentum_20d": "momentum_20d",
    "momentum_60d": "momentum_60d",
    "reversal": "reversal_5d",
    "reversal_5d": "reversal_5d",
    "volatility": "volatility_20d",
    "volatility_20d": "volatility_20d",
    "turnover": "turnover_20d",
    "turnover_20d": "turnover_20d",
    "amount_zscore": "amount_zscore_20d",
    "amount_zscore_20d": "amount_zscore_20d",
}


def _map_strategy_factor(strategy_id: str) -> str | None:
    """Resolve strategy_id to a built-in factor name."""
    sid = strategy_id.lower().strip()
    if sid in _STRATEGY_FACTORS:
        return _STRATEGY_FACTORS[sid]
    # Fuzzy match
    for key, value in _STRATEGY_FACTORS.items():
        if key in sid or sid in key:
            return value
    return None


def _factor_registry_root(lake: DataLake) -> Path:
    return lake.root.parent / "factors"


def _today_yyyymmdd() -> str:
    return datetime.now(tz=SHANGHAI_TZ).strftime("%Y%m%d")


def _requested_symbols(input_data: dict[str, Any]) -> list[str]:
    raw_symbols: list[Any] = []
    symbols_value = input_data.get("symbols", [])
    if isinstance(symbols_value, list):
        raw_symbols.extend(symbols_value)
    elif symbols_value:
        raw_symbols.append(symbols_value)
    for alias in ("symbol", "code", "universe"):
        value = input_data.get(alias)
        if value:
            raw_symbols.append(value)

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
