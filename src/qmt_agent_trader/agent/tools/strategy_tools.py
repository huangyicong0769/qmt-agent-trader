"""Strategy tools: create_strategy_spec, generate_strategy_code, run_backtest,
and report tool: generate_research_report."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import SHANGHAI_TZ, new_id, shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.loader import static_check_strategy_file
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategyKind,
    StrategySource,
    StrategySpec,
    strategy_spec_from_agent_spec,
)
from qmt_agent_trader.strategy.registry import StrategyRegistry

_sandbox: CodeSandbox | None = None
_store: ExperimentStore | None = None
_lake: DataLake | None = None
_sandbox_var: ContextVar[CodeSandbox | None] = ContextVar("strategy_tool_sandbox", default=None)
_store_var: ContextVar[ExperimentStore | None] = ContextVar("strategy_tool_store", default=None)
_lake_var: ContextVar[DataLake | None] = ContextVar("strategy_tool_lake", default=None)


def wire(sandbox: CodeSandbox, store: ExperimentStore, lake: DataLake) -> None:
    global _sandbox, _store, _lake
    _sandbox = sandbox
    _store = store
    _lake = lake


def _get_sandbox() -> CodeSandbox | None:
    return _sandbox_var.get() or _sandbox


def _get_store() -> ExperimentStore | None:
    return _store_var.get() or _store


def _get_lake() -> DataLake | None:
    return _lake_var.get() or _lake


def _with_deps(
    deps: AgentToolDependencies,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    sandbox_token = _sandbox_var.set(deps.sandbox)
    store_token = _store_var.set(deps.experiment_store)
    lake_token = _lake_var.set(deps.data_lake)
    try:
        return fn(input_data, context)
    finally:
        _lake_var.reset(lake_token)
        _store_var.reset(store_token)
        _sandbox_var.reset(sandbox_token)


# ── create_strategy_spec ────────────────────────────────────────────────────


def _create_strategy_spec(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    strategy_idea = input_data.get("strategy_idea", "")
    selected_factors = input_data.get("selected_factors", [])
    universe = input_data.get("universe", "stock_etf")
    rebalance_freq = input_data.get("rebalance_frequency", "daily")
    constraints = input_data.get("constraints", {})

    strategy_id = new_id("strat")
    kind = (
        StrategyKind.ETF_TREND
        if "etf" in str(strategy_idea).lower() and "trend" in str(strategy_idea).lower()
        else StrategyKind.FACTOR_RANK_LONG_ONLY
    )
    spec = StrategySpec(
        strategy_id=strategy_id,
        name=strategy_idea[:60] or "candidate_strategy",
        version="0.1.0",
        description=strategy_idea,
        kind=kind,
        source=StrategySource.AGENT_GENERATED,
        universe=universe,
        factors=[{"factor_id": str(item), "weight": 1.0} for item in selected_factors],
        portfolio={"method": "equal_weight_top_n", "top_n": 20},
        rebalance={"frequency": rebalance_freq},
        risk_constraints=constraints,
        execution={
            "signal_timing": "after_close",
            "execution_timing": "next_open",
            "execution_delay_days": 1,
            "slippage_bps": 5.0,
        },
    )
    return {"status": "created", "strategy_spec": spec.model_dump(mode="json")}


create_strategy_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_strategy_spec",
        description="将策略想法和候选因子组合转成结构化 strategy spec。",
        input_schema={
            "type": "object",
            "properties": {
                "strategy_idea": {"type": "string"},
                "selected_factors": {"type": "array", "items": {"type": "string"}},
                "universe": {"type": "string"},
                "rebalance_frequency": {"type": "string"},
                "constraints": {"type": "object"},
            },
            "required": ["strategy_idea", "selected_factors"],
            "additionalProperties": False,
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_create_strategy_spec,
)

# ── generate_strategy_code ──────────────────────────────────────────────────


def _generate_strategy_code(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("strategy_spec", {})
    spec = strategy_spec_from_agent_spec(spec_data)
    name = spec_data.get("name", "candidate")
    strategy_id = spec.strategy_id
    factors = [factor.factor_id for factor in spec.factors]
    warnings: list[str] = []

    if not factors:
        warnings.append("no factors selected; strategy will not generate signals")

    strategy_code = _render_strategy_code(name, spec)
    test_code = _render_strategy_test_code(name)

    sb = _get_sandbox()
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
            "strategy_id": strategy_id,
            "strategy_spec": spec.model_dump(mode="json"),
            "warnings": warnings,
        }
    except Exception as exc:
        return {"status": "STATIC_CHECK_FAILED", "message": str(exc), "warnings": [str(exc)]}


generate_strategy_code_tool: AgentTool = tool(
    ToolSpec(
        name="generate_strategy_code",
        description="根据 strategy spec 生成候选策略代码和测试。",
        input_schema={
            "type": "object",
            "properties": {"strategy_spec": {"type": "object"}},
            "required": ["strategy_spec"],
            "additionalProperties": False,
        },
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_strategy_code,
)

# ── list_strategy_candidates ────────────────────────────────────────────────


def _list_strategy_candidates(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    sb = _get_sandbox()
    if sb is None:
        return {"status": "NOT_IMPLEMENTED", "message": "sandbox not wired"}

    query = str(input_data.get("query") or "").strip()
    strategy_registry = _strategy_registry()
    root = sb.generated_root / "strategies"
    candidates: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/strategy.py")):
        strategy_id = path.parent.name
        tests_path = path.with_name("test_strategy.py")
        if query and query not in strategy_id and query not in str(path):
            continue
        candidates.append(
            {
                "strategy_id": strategy_id,
                "status": _registered_status(strategy_registry, strategy_id) or "draft",
                "code_path": str(path),
                "tests_path": str(tests_path) if tests_path.exists() else None,
                "saved": strategy_registry.get_strategy(strategy_id) is not None,
                "saved_in_registry": strategy_registry.get_strategy(strategy_id) is not None,
                "report_paths": _registered_reports(strategy_registry, strategy_id),
                "approval_file": _registered_approval(strategy_registry, strategy_id),
            }
        )
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        strategy_id = path.stem
        tests_path = path.with_name(f"test_{path.name}")
        if query and query not in strategy_id and query not in str(path):
            continue
        candidates.append(
            {
                "strategy_id": strategy_id,
                "status": _registered_status(strategy_registry, strategy_id) or "draft",
                "code_path": str(path),
                "tests_path": str(tests_path) if tests_path.exists() else None,
                "saved": strategy_registry.get_strategy(strategy_id) is not None,
                "saved_in_registry": strategy_registry.get_strategy(strategy_id) is not None,
                "report_paths": _registered_reports(strategy_registry, strategy_id),
                "approval_file": _registered_approval(strategy_registry, strategy_id),
            }
        )
    return {
        "status": "ok",
        "query": query or None,
        "count": len(candidates),
        "strategies": candidates,
    }


list_strategy_candidates_tool: AgentTool = tool(
    ToolSpec(
        name="list_strategy_candidates",
        description="查询 Agent 已生成的策略候选、代码路径和草稿状态。",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_list_strategy_candidates,
)

# ── save_strategy_candidate ─────────────────────────────────────────────────


def _save_strategy_candidate(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("strategy_spec")
    if not isinstance(spec_data, dict):
        return {"status": "error", "message": "strategy_spec is required"}
    spec = strategy_spec_from_agent_spec(spec_data)
    code_path = str(input_data.get("code_path") or "")
    if not code_path:
        return {"status": "error", "message": "code_path is required"}
    tests_path = str(input_data.get("tests_path") or "") or None
    registry = _strategy_registry()
    existing = registry.get_strategy(spec.strategy_id)
    if existing is not None:
        return {
            "status": "already_saved",
            "strategy_id": spec.strategy_id,
            "saved_strategy": existing.model_dump(mode="json"),
        }

    saved = SavedStrategy(
        strategy_id=spec.strategy_id,
        name=spec.name,
        version=spec.version,
        source=StrategySource.AGENT_GENERATED,
        status=ApprovalStatus.GENERATED_BY_LLM,
        spec=spec,
        implementation_ref=f"file:{code_path}",
        code_path=code_path,
        tests_path=tests_path,
        created_by="agent",
    )
    stored = registry.save_candidate(saved)
    return {
        "status": "saved",
        "strategy_id": spec.strategy_id,
        "saved_strategy": stored.model_dump(mode="json"),
        "review_required": True,
        "live_trading_allowed": False,
    }


save_strategy_candidate_tool: AgentTool = tool(
    ToolSpec(
        name="save_strategy_candidate",
        description=(
            "将已生成并通过静态检查的 strategy spec/code_path 保存到策略注册表。"
            "保存后仍是 GENERATED_BY_LLM/REVIEW_REQUIRED，不能直接实盘。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "strategy_spec": {"type": "object"},
                "code_path": {"type": "string"},
                "tests_path": {"type": "string"},
            },
            "required": ["strategy_spec", "code_path"],
            "additionalProperties": False,
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_save_strategy_candidate,
)

# ── run_strategy_static_checks ──────────────────────────────────────────────


def _run_strategy_static_checks(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    code_path = input_data.get("code_path")
    if not code_path:
        return {"status": "FAILED", "issues": ["code_path is required"]}
    issues = static_check_strategy_file(Path(str(code_path)))
    return {
        "status": "PASSED" if not issues else "FAILED",
        "issues": issues,
        "code_path": str(code_path),
    }


run_strategy_static_checks_tool: AgentTool = tool(
    ToolSpec(
        name="run_strategy_static_checks",
        description=(
            "检查候选策略代码是否包含未来函数、危险 import、broker/gateway "
            "或 live trading 调用。"
        ),
        input_schema={
            "type": "object",
            "properties": {"code_path": {"type": "string"}},
            "required": ["code_path"],
            "additionalProperties": False,
        },
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=True,
    ),
    fn=_run_strategy_static_checks,
)

# ── run_backtest ────────────────────────────────────────────────────────────


def _run_backtest(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_IMPLEMENTED", "message": "data lake not wired"}

    strategy_id = input_data.get("strategy_id", "")
    factor_name = input_data.get("factor_name", "")
    spec_data = input_data.get("strategy_spec")
    strategy_spec = (
        strategy_spec_from_agent_spec(spec_data)
        if isinstance(spec_data, dict)
        else None
    )
    if strategy_spec is None and strategy_id:
        saved_strategy = _strategy_registry().get_strategy(str(strategy_id))
        if saved_strategy is not None:
            strategy_spec = saved_strategy.spec
    if strategy_spec is not None:
        strategy_id = strategy_id or strategy_spec.strategy_id
        if not factor_name and strategy_spec.factors:
            factor_name = strategy_spec.factors[0].factor_id
    start_date = input_data.get("start_date", "20200101")
    end_date = input_data.get("end_date", _today_yyyymmdd())
    initial_cash = float(input_data.get("initial_cash", 1_000_000))
    top_n = int(input_data.get("top_n", strategy_spec.portfolio.top_n if strategy_spec else 20))
    symbols = _requested_symbols(input_data)
    code_path = str(input_data.get("code_path") or "")
    if code_path:
        issues = static_check_strategy_file(Path(code_path))
        if issues:
            return {"status": "STATIC_CHECK_FAILED", "issues": issues, "code_path": code_path}

    # Resolve factor_name from strategy_id if not provided
    if not factor_name and strategy_id:
        factor_name = _map_strategy_factor(strategy_id)
    if not factor_name:
        return {
            "status": "error",
            "message": "必须提供 factor_name、strategy_spec 或可映射到因子的 strategy_id。"
        }

    registry_root = _factor_registry_root(lake)
    factor_registry = FactorRegistry(registry_root)
    saved = factor_registry.get_factor(factor_name)
    if saved is None:
        return {
            "status": "FACTOR_NOT_FOUND",
            "message": (
                f"factor '{factor_name}' is not an exact saved factor_id/name. "
                "Call list_saved_factors and use an exact factor_id."
            ),
            "candidates": [
                {
                    "factor_id": item.factor_id,
                    "name": item.name,
                    "status": item.status,
                    "created_by": item.created_by,
                    "created_at": item.created_at,
                }
                for item in factor_registry.find_factors(factor_name, include_builtins=True)
            ],
        }
    factor_name = saved.factor_id
    if strategy_spec is None:
        strategy_spec = StrategySpec(
            strategy_id=strategy_id or f"factor_{factor_name}",
            name=f"Factor baseline: {factor_name}",
            kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
            factors=[{"factor_id": factor_name}],
            portfolio={"top_n": top_n},
        )
    result = run_strategy_backtest(
        lake,
        _strategy_registry(),
        StrategyBacktestConfig(
            strategy_id=strategy_spec.strategy_id,
            strategy_spec=strategy_spec,
            factor_name=factor_name,
            start_date=start_date,
            end_date=end_date,
            universe=str(input_data.get("universe") or strategy_spec.universe),
            initial_cash=initial_cash,
            top_n=top_n,
            max_single_position_pct=strategy_spec.portfolio.max_single_position_pct,
            slippage_bps=strategy_spec.execution.slippage_bps,
            execution_delay_days=strategy_spec.execution.execution_delay_days,
            symbols=symbols,
        ),
        reports_dir=Path("reports/research"),
    )
    payload = result.model_dump(mode="json")
    if result.status == "completed":
        payload.update(
            {
                "factor_name": factor_name,
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "actual_data_start": result.data_window.get("actual_start"),
                "actual_data_end": result.data_window.get("actual_end"),
                "data_freshness": result.data_window.get("data_freshness"),
                "cache_hit": False,
            }
        )
    return payload


run_backtest_tool: AgentTool = tool(
    ToolSpec(
        name="run_backtest",
        description=(
            "运行因子排名策略的基线回测。返回 total_return, sharpe, max_drawdown, "
            "turnover, trade_count。必须提供 factor_name、strategy_spec 或可映射到因子的 "
            "strategy_id。传入多因子 strategy_spec 时，当前 baseline adapter 只用第一个"
            "因子，并在 warnings/adapter_limitations 中说明。"
            " 内置因子: momentum_20d, momentum_60d, reversal_5d, volatility_20d,"
            " turnover_20d, amount_zscore_20d"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "factor_name": {"type": "string"},
                "strategy_id": {"type": "string"},
                "strategy_spec": {"type": "object"},
                "code_path": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "symbol": {"type": "string"},
                "code": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "initial_cash": {"type": "number"},
                "top_n": {"type": "integer"},
            },
            "anyOf": [
                {"required": ["factor_name"]},
                {"required": ["strategy_spec"]},
                {"required": ["strategy_id"]},
            ],
            "additionalProperties": False,
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

    sb = _get_sandbox()
    reports_root = sb.generated_root / "reports" if sb else Path("reports/research")
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"{exp_id}.md"

    store = _get_store()
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
        input_schema={
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "include_sections": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_research_report,
)


def build_strategy_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    definitions: list[
        tuple[AgentTool, Callable[[dict[str, Any], ToolContext], dict[str, Any]]]
    ] = [
        (create_strategy_spec_tool, _create_strategy_spec),
        (generate_strategy_code_tool, _generate_strategy_code),
        (list_strategy_candidates_tool, _list_strategy_candidates),
        (save_strategy_candidate_tool, _save_strategy_candidate),
        (run_strategy_static_checks_tool, _run_strategy_static_checks),
        (run_backtest_tool, _run_backtest),
        (generate_research_report_tool, _generate_research_report),
    ]
    return [_bind_tool(deps, existing, fn) for existing, fn in definitions]


def _bind_tool(
    deps: AgentToolDependencies,
    existing: AgentTool,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
) -> AgentTool:
    return tool(
        existing.spec,
        fn=lambda input_data, context: _with_deps(deps, fn, input_data, context),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _render_strategy_code(name: str, spec: StrategySpec) -> str:
    name.replace(" ", "_").replace("-", "_").lower()
    spec_literal = pformat(spec.model_dump(mode="json"), width=100)
    return f'''"""Candidate strategy: {name}.
Auto-generated by Agent. REVIEW_REQUIRED before promotion.
"""

import pandas as pd

from qmt_agent_trader.strategy.base import StrategyContext
from qmt_agent_trader.strategy.portfolio import equal_weight_top_n_from_scores

STRATEGY_SPEC = {spec_literal}
FACTORS = [item["factor_id"] for item in STRATEGY_SPEC.get("factors", [])]
TOP_N = int(STRATEGY_SPEC.get("portfolio", {{}}).get("top_n", 20))
MAX_SINGLE_POSITION_PCT = float(
    STRATEGY_SPEC.get("portfolio", {{}}).get("max_single_position_pct", 0.10)
)
CASH_BUFFER_PCT = float(STRATEGY_SPEC.get("portfolio", {{}}).get("cash_buffer_pct", 0.02))


def generate_signals(context: StrategyContext) -> pd.DataFrame:
    """Generate long-only target weights from configured factor columns."""
    data = context.factors if isinstance(context.factors, pd.DataFrame) else context.bars
    if not isinstance(data, pd.DataFrame) or data.empty:
        return _empty_signals()
    if "symbol" not in data.columns:
        raise ValueError("strategy input must include symbol")
    missing = [factor for factor in FACTORS if factor not in data.columns]
    if missing:
        raise ValueError(f"missing factor columns: {{missing}}")
    scored = data.drop_duplicates("symbol", keep="last").copy()
    scored["score"] = 0.0
    for item in STRATEGY_SPEC.get("factors", []):
        factor_id = item["factor_id"]
        weight = float(item.get("weight", 1.0))
        values = pd.to_numeric(scored[factor_id], errors="coerce")
        std = float(values.std(ddof=0))
        mean = float(values.mean())
        normalized = values - mean if std <= 0 else (values - mean) / std
        if bool(item.get("ascending", False)):
            normalized = -normalized
        scored["score"] = scored["score"] + normalized.fillna(0.0) * weight
    signals = equal_weight_top_n_from_scores(
        scored,
        top_n=TOP_N,
        max_single_position_pct=MAX_SINGLE_POSITION_PCT,
        cash_buffer_pct=CASH_BUFFER_PCT,
        score_column="score",
    )
    signal_date = context.as_of_date
    signals.insert(1, "signal_date", signal_date)
    scores = scored.set_index("symbol")["score"]
    signals.insert(2, "score", signals["symbol"].map(scores))
    return signals[["symbol", "signal_date", "score", "target_weight", "reason"]]


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "signal_date", "score", "target_weight", "reason"])
'''


def _render_strategy_test_code(name: str) -> str:
    return f'''"""Tests for candidate strategy: {name}."""

import importlib.util
from pathlib import Path

import pandas as pd
from qmt_agent_trader.strategy.base import StrategyContext


MODULE_PATH = Path(__file__).with_name("strategy.py")
SPEC = importlib.util.spec_from_file_location("candidate_strategy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
generate_signals = MODULE.generate_signals


def test_empty_data():
    context = StrategyContext(as_of_date="20240102", universe="stock_etf", bars=pd.DataFrame())
    result = generate_signals(context)
    assert result.empty


def test_generates_signals():
    factor_id = MODULE.FACTORS[0] if MODULE.FACTORS else "score"
    data = pd.DataFrame({{
        "symbol": ["A", "B", "C"],
        "close": [10, 20, 30],
        factor_id: [0.9, 0.5, 0.1],
    }})
    context = StrategyContext(as_of_date="20240102", universe="stock_etf", bars=data, factors=data)
    result = generate_signals(context)
    assert "symbol" in result.columns
    assert "target_weight" in result.columns
    assert "score" in result.columns
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


def _strategy_registry() -> StrategyRegistry:
    lake = _get_lake()
    root = lake.root.parent / "strategies" if lake is not None else Path("data/strategies")
    return StrategyRegistry(root)


def _registered_status(registry: StrategyRegistry, strategy_id: str) -> str | None:
    saved = registry.get_strategy(strategy_id)
    return saved.status.value if saved is not None else None


def _registered_reports(registry: StrategyRegistry, strategy_id: str) -> list[str]:
    saved = registry.get_strategy(strategy_id)
    return saved.report_paths if saved is not None else []


def _registered_approval(registry: StrategyRegistry, strategy_id: str) -> str | None:
    saved = registry.get_strategy(strategy_id)
    return saved.approval_file if saved is not None else None


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
