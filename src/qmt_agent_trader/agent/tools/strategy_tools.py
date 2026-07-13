"""Strategy tools: create_strategy_spec, generate_strategy_code, run_backtest,
and report tool: generate_research_report."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict
from collections.abc import Callable
from contextvars import ContextVar
from datetime import date, datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, cast

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox, generated_identity_segment
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    ExecutionStatus,
    RecommendationStatus,
)
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import SHANGHAI_TZ, new_id, shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.persistence.artifacts import ArtifactMetadata, artifact_store_for_root
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.cache import ContentAddressedCache
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.paths import PersistencePaths
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.adapter_capabilities import (
    validate_factor_rank_adapter_spec,
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
from qmt_agent_trader.universe.builtins import broad_universe_spec
from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.registry import UniverseRegistry, registry_root_from_payload
from qmt_agent_trader.universe.resolver import UniverseResolver

_sandbox: CodeSandbox | None = None
_store: ExperimentStore | None = None
_lake: DataLake | None = None
_sandbox_var: ContextVar[CodeSandbox | None] = ContextVar("strategy_tool_sandbox", default=None)
_store_var: ContextVar[ExperimentStore | None] = ContextVar("strategy_tool_store", default=None)
_lake_var: ContextVar[DataLake | None] = ContextVar("strategy_tool_lake", default=None)
_cache_var: ContextVar[ContentAddressedCache | None] = ContextVar(
    "strategy_tool_cache", default=None
)
BROAD_UNIVERSE_MIN_SYMBOLS = 500
BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v2"


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


def _factor_registry(lake: DataLake, root: Path | None = None) -> FactorRegistry:
    return FactorRegistry(
        root or _factor_registry_root(lake),
        lock_manager=lake.lock_manager,
        atomic_store=AtomicFileStore(lake.lock_manager),
    )


def _with_deps(
    deps: AgentToolDependencies,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    sandbox_token = _sandbox_var.set(deps.sandbox)
    store_token = _store_var.set(deps.experiment_store)
    lake_token = _lake_var.set(deps.data_lake)
    cache_token = _cache_var.set(deps.cache)
    try:
        return fn(input_data, context)
    finally:
        _cache_var.reset(cache_token)
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
    constraints = constraints if isinstance(constraints, dict) else {}

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
        factors=_factor_legs_from_selected(selected_factors, constraints),
        portfolio=_portfolio_from_constraints(constraints),
        rebalance={"frequency": rebalance_freq},
        risk_constraints=constraints,
        execution=_execution_from_constraints(constraints),
    )
    return {
        "status": "created",
        "strategy_spec": spec.model_dump(mode="json"),
        "warnings": [],
        "saved_in_registry": False,
        "research_only": True,
        "live_trading_allowed": False,
        "suggested_next_tools": [
            "generate_strategy_code",
            "run_strategy_static_checks",
            "save_strategy_candidate",
            "save_strategy_spec_draft",
            "run_backtest",
        ],
    }


create_strategy_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_strategy_spec",
        description=(
            "将策略想法和候选因子组合转成结构化 strategy spec，并保留 factor "
            "weights、ascending/lower_is_better 方向、portfolio 和 execution 约束。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "strategy_idea": {"type": "string"},
                "selected_factors": {
                    "type": "array",
                    "items": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                },
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
        run_segment = generated_identity_segment(context.run_id)
        version_segment = generated_identity_segment(spec.version)
        run_root = f"strategies/drafts/{strategy_id}/{version_segment}/{run_segment}"
        code_path = sb.write_candidate_file(
            f"{run_root}/strategy.py",
            strategy_code,
            artifact_id=f"strategy:{strategy_id}:{spec.version}:{context.run_id}:implementation",
            related_run_id=context.run_id,
            related_strategy_id=strategy_id,
        )
        tests_path = sb.write_candidate_file(
            f"{run_root}/test_strategy.py",
            test_code,
            artifact_id=f"strategy:{strategy_id}:{spec.version}:{context.run_id}:tests",
            related_run_id=context.run_id,
            related_strategy_id=strategy_id,
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
        relative_parts = path.relative_to(root).parts
        strategy_id = (
            relative_parts[1]
            if len(relative_parts) >= 3 and relative_parts[0] == "drafts"
            else path.parent.name
        )
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
        if existing.source == StrategySource.AGENT_GENERATED and (
            existing.implementation_ref == "spec:draft" or existing.code_path is None
        ):
            stored = registry.attach_generated_implementation(
                spec.strategy_id,
                spec=spec,
                code_path=code_path,
                tests_path=tests_path,
            )
            return {
                "status": "updated",
                "registry_action": "attached_generated_implementation",
                "strategy_id": spec.strategy_id,
                "saved_strategy": stored.model_dump(mode="json"),
                "review_required": True,
                "live_trading_allowed": False,
            }
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


# ── save_strategy_spec_draft ──────────────────────────────────────────────────


def _save_strategy_spec_draft(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("strategy_spec")
    if not isinstance(spec_data, dict):
        return {"status": "error", "message": "strategy_spec is required"}
    spec = strategy_spec_from_agent_spec(spec_data)
    registry = _strategy_registry()
    existing = registry.get_strategy(spec.strategy_id)
    if existing is not None:
        return {
            "status": "already_saved",
            "strategy_id": spec.strategy_id,
            "saved_in_registry": True,
            "saved_strategy": existing.model_dump(mode="json"),
            "research_only": True,
            "live_trading_allowed": False,
            "review_required": True,
        }

    saved = SavedStrategy(
        strategy_id=spec.strategy_id,
        name=spec.name,
        version=spec.version,
        source=StrategySource.AGENT_GENERATED,
        status=ApprovalStatus.GENERATED_BY_LLM,
        spec=spec,
        implementation_ref="spec:draft",
        code_path=None,
        tests_path=None,
        created_by="agent",
    )
    stored = registry.save_candidate(saved)
    return {
        "status": "saved",
        "strategy_id": spec.strategy_id,
        "saved_in_registry": True,
        "code_path": None,
        "research_only": True,
        "live_trading_allowed": False,
        "review_required": True,
        "saved_strategy": stored.model_dump(mode="json"),
    }


save_strategy_spec_draft_tool: AgentTool = tool(
    ToolSpec(
        name="save_strategy_spec_draft",
        description=(
            "将 research-only strategy spec 草稿保存到策略注册表，不要求已生成 code_path。"
            "保存后仍需人工 review，不能直接实盘。"
        ),
        input_schema={
            "type": "object",
            "properties": {"strategy_spec": {"type": "object"}},
            "required": ["strategy_spec"],
            "additionalProperties": False,
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_save_strategy_spec_draft,
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
            "检查候选策略代码是否包含未来函数、危险 import、broker/gateway 或 live trading 调用。"
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
    saved_strategy: SavedStrategy | None = None
    strategy_spec_result = _parse_backtest_strategy_spec(spec_data, input_data)
    if isinstance(strategy_spec_result, dict):
        return strategy_spec_result
    strategy_spec = strategy_spec_result
    if strategy_spec is None and strategy_id:
        saved_strategy = _strategy_registry().get_strategy(str(strategy_id))
        if saved_strategy is not None:
            strategy_spec = saved_strategy.spec
        elif not factor_name:
            return {
                "status": "STRATEGY_NOT_FOUND",
                "strategy_id": str(strategy_id),
                "message": (
                    "strategy_id not found in StrategyRegistry; pass strategy_spec "
                    "or save the spec draft first"
                ),
                "suggested_next_tools": ["save_strategy_spec_draft", "list_strategy_candidates"],
                "research_only": True,
                "live_trading_allowed": False,
            }
    if strategy_spec is not None:
        strategy_id = strategy_id or strategy_spec.strategy_id
        if not factor_name and strategy_spec.factors:
            factor_name = strategy_spec.factors[0].factor_id
    code_path = str(input_data.get("code_path") or "")
    if strategy_spec is not None:
        capability_issues = validate_factor_rank_adapter_spec(
            strategy_spec,
            code_path=code_path or None,
        )
        if capability_issues:
            generated_code = any(issue.field == "code_path" for issue in capability_issues)
            return {
                "status": "BLOCKED",
                "reason": (
                    "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"
                    if generated_code
                    else "UNSUPPORTED_STRATEGY_SEMANTICS"
                ),
                "unsupported_fields": [issue.field for issue in capability_issues],
                "capability_issues": [asdict(issue) for issue in capability_issues],
                "execution_backend": "factor_rank_baseline_adapter",
                "research_only": True,
                "live_trading_allowed": False,
            }
    start_date = input_data.get("start_date", "20200101")
    end_date = input_data.get("end_date", _today_yyyymmdd())
    initial_cash = float(input_data.get("initial_cash", 1_000_000))
    top_n = int(input_data.get("top_n", strategy_spec.portfolio.top_n if strategy_spec else 20))
    symbols = _requested_symbols(input_data)
    universe_state = _resolve_backtest_universe(
        lake,
        input_data,
        strategy_spec=strategy_spec,
        saved_strategy=saved_strategy,
        symbols=symbols,
        start_date=str(start_date),
        end_date=str(end_date),
    )
    if universe_state["status"] != "OK":
        payload = dict(universe_state["payload"])
        if strategy_spec is not None:
            payload.setdefault("strategy_id", strategy_spec.strategy_id)
        elif strategy_id:
            payload.setdefault("strategy_id", str(strategy_id))
        if saved_strategy is not None and not saved_strategy.code_path:
            warnings = list(payload.get("warnings") or [])
            warning = "strategy has no generated code; backtest used canonical adapter"
            if warning not in warnings:
                warnings.append(warning)
            payload["warnings"] = warnings
            payload["saved_in_registry"] = True
            payload["generated_code"] = False
            payload["static_checks"] = "NOT_RUN"
        return payload
    symbols = universe_state["symbols"]
    symbols_by_date = universe_state["symbols_by_date"]
    universe_info = universe_state["universe_info"]
    resolved_universe = universe_state["resolved_universe"]
    if not factor_name:
        return {
            "status": "INVALID_REQUEST",
            "message": "必须提供 factor_name、strategy_spec 或已保存的 strategy_id。",
            "suggested_next_tools": ["create_strategy_spec", "save_strategy_spec_draft"],
        }

    registry_root = _factor_registry_root(lake)
    factor_registry = _factor_registry(lake, registry_root)
    requested_factor_ids = (
        [factor.factor_id for factor in strategy_spec.factors]
        if strategy_spec is not None and strategy_spec.factors
        else [str(factor_name)]
    )
    missing_factor_ids = [
        item for item in requested_factor_ids if factor_registry.get_factor(item) is None
    ]
    if missing_factor_ids:
        return {
            "status": "FACTOR_NOT_FOUND",
            "message": (
                f"factor '{missing_factor_ids[0]}' is not an exact saved factor_id/name. "
                "Call list_saved_factors and use an exact factor_id."
            ),
            "missing_factor_ids": missing_factor_ids,
            "candidates": [
                {
                    "factor_id": item.factor_id,
                    "name": item.name,
                    "status": item.status,
                    "created_by": item.created_by,
                    "created_at": item.created_at,
                }
                for item in factor_registry.find_factors(
                    missing_factor_ids[0],
                    include_builtins=True,
                )
            ],
        }
    saved = factor_registry.get_factor(str(factor_name))
    factor_name = saved.factor_id if saved is not None else requested_factor_ids[0]
    if strategy_spec is None:
        strategy_spec = StrategySpec(
            strategy_id=strategy_id or f"factor_{factor_name}",
            name=f"Factor baseline: {factor_name}",
            kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
            factors=[{"factor_id": factor_name}],
            portfolio={"top_n": top_n},
        )
    single_factor = strategy_spec.factors[0] if len(strategy_spec.factors) == 1 else None
    config = StrategyBacktestConfig(
        strategy_id=strategy_spec.strategy_id,
        strategy_spec=strategy_spec,
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        universe=str(universe_info["universe_effective"]),
        initial_cash=initial_cash,
        top_n=top_n,
        max_single_position_pct=strategy_spec.portfolio.max_single_position_pct,
        slippage_bps=strategy_spec.execution.slippage_bps,
        execution_delay_days=strategy_spec.execution.execution_delay_days,
        rebalance_frequency=strategy_spec.rebalance.frequency,
        min_turnover_threshold=strategy_spec.rebalance.min_turnover_threshold,
        rank_buffer=strategy_spec.rebalance.rank_buffer,
        cash_buffer_pct=strategy_spec.portfolio.cash_buffer_pct,
        lower_is_better=bool(single_factor and single_factor.ascending),
        symbols=symbols,
        symbols_by_date=symbols_by_date,
        universe_mode=cast(
            Literal["snapshot", "rolling"],
            str(universe_info.get("universe_mode") or "snapshot"),
        ),
    )
    cost_estimate = _backtest_cost_estimate(config)
    timeout_seconds_used = _backtest_timeout_seconds_for_call(input_data, context)
    cache_key = _backtest_cache_key(
        lake,
        config=config,
        factor_name=factor_name,
        requested_factor_ids=requested_factor_ids,
    )
    cached = _get_cached_backtest(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        cached["timeout_seconds_used"] = timeout_seconds_used
        cached["cost_estimate"] = cost_estimate
        cached.update(_universe_evidence_payload(universe_info, symbols, resolved_universe))
        return _with_backtest_evidence_status(cached)
    try:
        result = run_strategy_backtest(
            lake,
            _strategy_registry(),
            config,
            reports_dir=PersistencePaths.from_settings(get_settings()).reports_root / "research",
        )
    except BacktestDataIntegrityError as exc:
        return {
            "status": "ERROR",
            "reason": "BACKTEST_DATA_INTEGRITY_ERROR",
            "error": {
                "code": exc.code,
                "trade_date": exc.trade_date,
                "symbols": list(exc.symbols),
                "field": exc.field,
                "message": exc.message,
            },
            "research_only": True,
            "live_trading_allowed": False,
        }
    except ValueError as exc:
        blocked = _blocked_backtest_from_value_error(
            exc,
            config=config,
            requested_factor_ids=requested_factor_ids,
        )
        if blocked is not None:
            return _with_backtest_evidence_status(blocked)
        raise
    payload = result.model_dump(mode="json")
    payload.update(_universe_evidence_payload(universe_info, symbols, resolved_universe))
    if saved_strategy is not None and not saved_strategy.code_path:
        warnings = list(payload.get("warnings") or [])
        warning = "strategy has no generated code; backtest used canonical adapter"
        if warning not in warnings:
            warnings.append(warning)
        payload["warnings"] = warnings
        payload["saved_in_registry"] = True
        payload["generated_code"] = False
        payload["static_checks"] = "NOT_RUN"
    if result.status == "completed":
        payload.update(
            {
                "factor_name": factor_name,
                "symbols": symbols,
                "universe_resolution": resolved_universe,
                "start_date": start_date,
                "end_date": end_date,
                "actual_data_start": result.data_window.get("actual_start"),
                "actual_data_end": result.data_window.get("actual_end"),
                "data_freshness": result.data_window.get("data_freshness"),
                "cache_hit": False,
                "timeout_seconds_used": timeout_seconds_used,
                "cost_estimate": cost_estimate,
            }
        )
        _put_cached_backtest(cache_key, payload)
    return _with_backtest_evidence_status(payload)


def _with_backtest_evidence_status(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostic_status = None
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostic_status = diagnostics.get("status")
    raw_status = str(payload.get("status") or "")
    enriched = dict(payload)
    enriched["execution_status"] = ExecutionStatus.OK.value
    enriched["raw_status"] = raw_status or None
    enriched["diagnostic_status"] = diagnostic_status
    if diagnostic_status == "FAIL":
        enriched.update(
            {
                "domain_status": DomainStatus.FAILED.value,
                "evidence_status": EvidenceStatus.INVALID.value,
                "recommendation_status": RecommendationStatus.DO_NOT_RECOMMEND.value,
                "message": payload.get("message") or "Backtest executed, but diagnostics failed.",
            }
        )
        return enriched
    if raw_status == "completed":
        if diagnostic_status in {"PASS", "OK"}:
            enriched.setdefault("domain_status", DomainStatus.OK.value)
            enriched.setdefault("evidence_status", EvidenceStatus.VALID.value)
        else:
            enriched.setdefault("domain_status", DomainStatus.UNKNOWN.value)
            enriched.setdefault("evidence_status", EvidenceStatus.UNKNOWN.value)
            warnings = list(enriched.get("warnings") or [])
            warning = "backtest_completed_without_explicit_diagnostic_pass"
            if warning not in warnings:
                warnings.append(warning)
            enriched["warnings"] = warnings
        enriched.setdefault("recommendation_status", RecommendationStatus.RESEARCH_ONLY.value)
        return enriched
    if raw_status in {"BLOCKED", "DATA_NOT_READY", "FACTOR_NOT_FOUND"}:
        enriched.setdefault("domain_status", DomainStatus.BLOCKED.value)
        enriched.setdefault("evidence_status", EvidenceStatus.BLOCKED.value)
        enriched.setdefault("recommendation_status", RecommendationStatus.BLOCKED.value)
        return enriched
    if raw_status in {"BACKTEST_FAILED", "STATIC_CHECK_FAILED"}:
        enriched.setdefault("domain_status", DomainStatus.FAILED.value)
        enriched.setdefault("evidence_status", EvidenceStatus.INVALID.value)
        enriched.setdefault("recommendation_status", RecommendationStatus.BLOCKED.value)
        return enriched
    enriched.setdefault("domain_status", DomainStatus.UNKNOWN.value)
    enriched.setdefault("evidence_status", EvidenceStatus.UNKNOWN.value)
    enriched.setdefault("recommendation_status", RecommendationStatus.UNKNOWN.value)
    return enriched


def _blocked_backtest_from_value_error(
    exc: ValueError,
    *,
    config: StrategyBacktestConfig,
    requested_factor_ids: list[str],
) -> dict[str, Any] | None:
    message = str(exc)
    if "missing required columns" not in message:
        return None
    factor_id = _factor_id_from_missing_columns_error(message) or config.factor_name
    missing_columns = _missing_columns_from_error(message)
    repair_tool = _repair_tool_for_missing_columns(missing_columns)
    return {
        "status": "BLOCKED",
        "reason": "MISSING_FACTOR_INPUTS",
        "message": message,
        "strategy_id": config.strategy_id,
        "factor_id": factor_id,
        "factor_ids": requested_factor_ids,
        "requested_factor_ids": requested_factor_ids,
        "missing_columns": missing_columns,
        "required_datasets": _datasets_for_missing_columns(missing_columns),
        "available_columns": [],
        "coverage_status": "NO_DATA",
        "missing_ranges": [
            {
                "start_date": config.start_date,
                "end_date": config.end_date,
                "symbols": config.symbols,
                "columns": missing_columns,
            }
        ],
        "datasets_used": [],
        "next_repair_tool": repair_tool,
        "suggested_repair": {
            "tool": repair_tool,
            "args": {
                "start_date": config.start_date,
                "end_date": config.end_date,
                "symbols": config.symbols,
                "columns": missing_columns,
            },
        },
        "research_only": True,
        "live_trading_allowed": False,
        "adapter_limitations": [],
        "diagnostics": {
            "status": "BLOCKED",
            "checks": [
                {
                    "name": "factor_input_columns",
                    "status": "BLOCKED",
                    "observed": missing_columns,
                    "threshold": "all required columns present",
                    "message": message,
                    "evidence_source": "blocked",
                }
            ],
        },
    }


def _factor_id_from_missing_columns_error(message: str) -> str | None:
    prefix = "factor '"
    if not message.startswith(prefix):
        return None
    return message[len(prefix) :].split("'", 1)[0]


def _missing_columns_from_error(message: str) -> list[str]:
    marker = "missing required columns:"
    if marker not in message:
        return []
    raw = message.split(marker, 1)[1].strip()
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _repair_tool_for_missing_columns(columns: list[str]) -> str:
    fundamentals = {
        "pb",
        "pe",
        "pe_ttm",
        "roe",
        "gross_margin",
        "debt_to_assets",
        "dv_ttm",
        "total_mv",
    }
    macro = {"pmi", "ppi", "cpi", "macro_cycle_score", "industry_value_added"}
    normalized = {column.lower() for column in columns}
    if normalized & fundamentals:
        return "run_tushare_fetch"
    if normalized & macro:
        return "run_tushare_fetch"
    return "run_tushare_fetch"


def _datasets_for_missing_columns(columns: list[str]) -> list[str]:
    daily_basic = {
        "pb",
        "pe",
        "pe_ttm",
        "dv_ttm",
        "total_mv",
        "circ_mv",
    }
    fina_indicator = {
        "roe",
        "gross_margin",
        "debt_to_assets",
        "current_ratio",
        "roic",
    }
    daily = {"open", "high", "low", "close", "vol", "volume", "amount", "turnover"}
    macro = {"pmi", "ppi", "cpi", "macro_cycle_score", "industry_value_added"}
    datasets: list[str] = []
    normalized = {column.lower() for column in columns}
    if normalized & daily_basic:
        datasets.append("tushare/daily_basic")
    if normalized & fina_indicator:
        datasets.append("tushare/fina_indicator")
    if normalized & daily:
        datasets.append("tushare/daily")
    if normalized & macro:
        datasets.append("macro_series")
    return datasets or ["custom_factor_inputs"]


def _parse_backtest_strategy_spec(
    spec_data: Any,
    input_data: dict[str, Any],
) -> StrategySpec | dict[str, Any] | None:
    if not isinstance(spec_data, dict):
        return None

    payload = dict(spec_data)
    factors = payload.get("factors") or payload.get("selected_factors") or payload.get("factor_ids")
    if factors is not None and "factors" not in payload:
        payload["factors"] = factors

    if not payload.get("strategy_id"):
        if not payload.get("factors"):
            return {
                "status": "INVALID_REQUEST",
                "message": (
                    "strategy_spec must include strategy_id or factors so run_backtest "
                    "can build a research-only temporary strategy."
                ),
                "missing_fields": ["strategy_id", "factors"],
            }
        payload["strategy_id"] = str(input_data.get("strategy_id") or new_id("strat"))
        payload.setdefault("name", f"Ad hoc factor strategy: {payload['strategy_id']}")
        payload.setdefault(
            "description",
            "Temporary research-only strategy_spec supplied to run_backtest.",
        )

    try:
        return strategy_spec_from_agent_spec(payload)
    except Exception as exc:
        return {
            "status": "INVALID_REQUEST",
            "message": f"invalid strategy_spec: {exc}",
            "strategy_spec_keys": sorted(str(key) for key in payload),
        }


run_backtest_tool: AgentTool = tool(
    ToolSpec(
        name="run_backtest",
        description=(
            "运行因子排名策略的 research-only 回测。必须提供 factor_name、strategy_spec "
            "或已保存的 strategy_id，并传入 symbols、universe_id、universe_spec "
            "或 broad universe_type；如确需默认大 universe，必须显式 "
            "allow_default_universe=true。返回 total_return, sharpe, "
            "max_drawdown, turnover, trade_count，并披露 symbols_source、symbols_count "
            "和 universe evidence。传入多因子 strategy_spec 时会按 factor weight 生成 "
            "composite score 并在 factor_ids/requested_factor_ids 中披露实际执行因子。"
            " 内置因子: momentum_20d, momentum_60d, reversal_5d, volatility_20d,"
            " turnover_20d, amount_zscore_20d, size_log_mktcap, pe_ttm_rank,"
            " pb_rank, dividend_yield, roe_rank, gross_margin_rank,"
            " debt_to_assets_rank。基本面因子会通过 PIT-safe factor input panel"
            " 拼接 daily_basic/fina_indicator 等字段；缺数据时返回具体 fetch guidance。"
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
                "universe": {"type": "string"},
                "universe_id": {"type": "string"},
                "universe_spec": {"type": "object"},
                "universe_mode": {"type": "string"},
                "universe_type": {"type": "string"},
                "as_of_date": {"type": "string"},
                "allow_default_universe": {"type": "boolean"},
                "universe_filters": {"type": "object"},
                "include_exclusions": {"type": "boolean"},
                "limit": {"type": "integer"},
                "rebalance_frequency": {"type": "string"},
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
    timeout_seconds_for_call=lambda input_data, context: _backtest_timeout_seconds_for_call(
        input_data,
        context,
    ),
)

# ── generate_research_report ────────────────────────────────────────────────


def _generate_research_report(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    exp_id = (
        input_data.get("experiment_id") or context.experiment_id or context.session_id or "unknown"
    )
    run_ids = input_data.get("run_ids", [])
    sections = input_data.get("include_sections", ["summary", "metrics"])

    sb = _get_sandbox()
    reports_root = (
        sb.generated_root / "reports"
        if sb
        else PersistencePaths.from_settings(get_settings()).reports_root / "research"
    )
    report_id = new_id("report")

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

    run_artifacts = [_load_run_artifact(str(run_id)) for run_id in run_ids]
    evidence_summary = [
        _report_evidence_summary(str(run_id), artifact)
        for run_id, artifact in zip(run_ids, run_artifacts, strict=False)
    ]

    if "summary" in sections:
        lines.extend(["## Summary", "", f"Run IDs: {', '.join(run_ids)}", ""])
    lines.extend(["## Evidence Status", ""])
    for run_id, artifact in zip(run_ids, run_artifacts, strict=False):
        if artifact is None:
            lines.append(f"- {run_id}: No run artifact found")
            continue
        status = _artifact_status(artifact)
        diagnostics = artifact.get("diagnostics", {})
        diagnostic_status = (
            diagnostics.get("status", "unknown") if isinstance(diagnostics, dict) else "unknown"
        )
        lines.append(f"- {run_id}: status={status}, diagnostics={diagnostic_status}")
    if not run_ids:
        lines.append("- No run IDs supplied")
    lines.append("")
    lines.extend(["## Effective Candidates / 有效候选", ""])
    lines.extend(_candidate_lines(run_ids, run_artifacts, group="effective"))
    lines.append("")
    lines.extend(["## Failed Candidates / 失败候选", ""])
    lines.extend(_candidate_lines(run_ids, run_artifacts, group="failed"))
    lines.append("")
    lines.extend(["## Blocked Candidates / 阻断候选", ""])
    lines.extend(_candidate_lines(run_ids, run_artifacts, group="blocked"))
    lines.append("")
    if "metrics" in sections:
        lines.extend(["## Metrics", ""])
        for artifact in run_artifacts:
            if not isinstance(artifact, dict):
                continue
            metrics = artifact.get("metrics")
            if isinstance(metrics, dict):
                lines.append(f"- {artifact.get('run_id', 'unknown')}: {metrics}")
        if not any(isinstance(item, dict) and item.get("metrics") for item in run_artifacts):
            lines.append("(no metrics artifact available)")
        lines.append("")
    if "limitations" in sections:
        lines.extend(["## Limitations", ""])
        limitation_lines = _report_limitations(run_artifacts)
        lines.extend(limitation_lines or ["- No explicit limitations captured in run artifacts"])
        lines.append("")
    if "data_gaps" in sections:
        lines.extend(["## Data Gaps / 数据缺口", ""])
        data_gap_lines = _report_data_gaps(run_artifacts)
        lines.extend(data_gap_lines or ["- No explicit data gaps captured in run artifacts"])
        lines.append("")
    lines.extend(["## Diagnostic Gaps / 诊断缺口", ""])
    diagnostic_gap_lines = _report_diagnostic_gaps(run_ids, run_artifacts)
    lines.extend(diagnostic_gap_lines or ["- No NOT_COMPUTED diagnostics captured"])
    lines.append("")
    lines.extend(["## Next Actions / 下一步动作", ""])
    lines.extend(_report_next_actions(run_ids, run_artifacts))
    lines.append("")
    if "lessons" in sections and experiment_meta.get("lessons"):
        lines.extend(["## Lessons Learned", ""])
        for lesson in experiment_meta["lessons"]:
            lines.append(f"- {lesson}")
        lines.append("")

    content = "\n".join(lines)
    if sb is not None and sb.artifact_store is not None:
        receipt = sb.artifact_store.create(
            f"reports/{exp_id}/{report_id}.md",
            content.encode("utf-8"),
            metadata=ArtifactMetadata(
                artifact_id=report_id,
                artifact_type="research_markdown_report",
                producer="agent.tools.strategy_tools.generate_research_report",
                related_run_id=str(run_ids[0]) if run_ids else context.run_id,
            ),
        )
    else:
        lake = _get_lake()
        manager = (
            lake.lock_manager
            if lake is not None
            else LockManager(PersistencePaths.from_settings(get_settings()).locks_root)
        )
        receipt = artifact_store_for_root(reports_root, lock_manager=manager).create(
            f"{exp_id}/{report_id}.md",
            content.encode("utf-8"),
            metadata=ArtifactMetadata(
                artifact_id=report_id,
                artifact_type="research_markdown_report",
                producer="agent.tools.strategy_tools.generate_research_report",
                related_run_id=str(run_ids[0]) if run_ids else context.run_id,
            ),
        )
    report_path = receipt.path
    return {
        "report_path": str(report_path),
        "manifest_path": str(receipt.manifest_path),
        "summary": lines[0],
        "evidence_summary": evidence_summary,
        "storage_status": {
            "status": "VERIFIED",
            "component": "artifact_store",
            "reason": None,
            "warnings": [],
            "repair_action": None,
        },
    }


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
    definitions: list[tuple[AgentTool, Callable[[dict[str, Any], ToolContext], dict[str, Any]]]] = [
        (create_strategy_spec_tool, _create_strategy_spec),
        (generate_strategy_code_tool, _generate_strategy_code),
        (list_strategy_candidates_tool, _list_strategy_candidates),
        (save_strategy_candidate_tool, _save_strategy_candidate),
        (save_strategy_spec_draft_tool, _save_strategy_spec_draft),
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
        timeout_seconds_for_call=getattr(existing, "timeout_seconds_for_call", None),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _factor_legs_from_selected(
    selected_factors: Any,
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_items = selected_factors if isinstance(selected_factors, list) else []
    weights = constraints.get("factor_weights")
    weights = weights if isinstance(weights, dict) else {}
    directions = constraints.get("factor_directions")
    directions = directions if isinstance(directions, dict) else {}

    legs: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            raw = dict(item)
            factor_id = str(raw.get("factor_id") or raw.get("name") or "").strip()
            if not factor_id:
                continue
            leg: dict[str, Any] = {"factor_id": factor_id}
            if "weight" in raw:
                leg["weight"] = float(raw["weight"])
            if "ascending" in raw:
                leg["ascending"] = bool(raw["ascending"])
            elif "direction" in raw:
                leg["ascending"] = _direction_is_ascending(raw["direction"])
            if raw.get("transform") is not None:
                leg["transform"] = str(raw["transform"])
        else:
            factor_id = str(item).strip()
            if not factor_id:
                continue
            leg = {"factor_id": factor_id}
        if factor_id in weights:
            leg["weight"] = float(weights[factor_id])
        if factor_id in directions:
            leg["ascending"] = _direction_is_ascending(directions[factor_id])
        leg.setdefault("weight", 1.0)
        leg.setdefault("ascending", False)
        legs.append(leg)
    return legs


def _direction_is_ascending(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"ascending", "asc", "lower_is_better", "low", "smaller_is_better"}


def _portfolio_from_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    portfolio = {"method": "equal_weight_top_n", "top_n": int(constraints.get("top_n", 20))}
    for key in ("max_single_position_pct", "cash_buffer_pct", "long_only"):
        if key in constraints:
            portfolio[key] = constraints[key]
    return portfolio


def _execution_from_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    execution: dict[str, Any] = {
        "signal_timing": "after_close",
        "execution_timing": "next_open",
        "execution_delay_days": int(constraints.get("execution_delay_days", 1)),
        "slippage_bps": float(constraints.get("slippage_bps", 5.0)),
    }
    if "cost_model" in constraints:
        execution["cost_model"] = str(constraints["cost_model"])
    return execution


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


def _load_run_artifact(run_id: str) -> dict[str, Any] | None:
    paths = PersistencePaths.from_settings(get_settings())
    for root in (paths.reports_root / "research", paths.reports_root / "backtests"):
        path = root / f"{run_id}.json"
        if not path.exists():
            continue
        lake = _get_lake()
        manager = (
            lake.lock_manager
            if lake is not None
            else LockManager(PersistencePaths.from_settings(get_settings()).locks_root)
        )
        store = artifact_store_for_root(root, lock_manager=manager)
        try:
            raw = store.read_verified(run_id, expected_relative_path=path.name)
            payload = json.loads(raw)
        except Exception as exc:
            return {
                "run_id": run_id,
                "status": "BLOCKED",
                "reason": "ARTIFACT_VERIFICATION_FAILED",
                "warnings": [str(exc)],
            }
        return payload if isinstance(payload, dict) else None
    return None


def _candidate_lines(
    run_ids: list[Any],
    artifacts: list[dict[str, Any] | None],
    *,
    group: str,
) -> list[str]:
    lines: list[str] = []
    for run_id, artifact in zip(run_ids, artifacts, strict=False):
        if _candidate_group(artifact) != group:
            continue
        lines.append(_candidate_line(str(run_id), artifact))
    return lines or ["- None"]


def _candidate_group(artifact: dict[str, Any] | None) -> str:
    if not isinstance(artifact, dict):
        return "blocked"
    status = _artifact_status(artifact)
    diagnostic_status = _diagnostic_status(artifact)
    if status.upper() in {
        "BLOCKED",
        "NO_DATA",
        "DATA_NOT_READY",
        "INVALID_REQUEST",
        "FACTOR_NOT_FOUND",
        "BACKTEST_FAILED",
    }:
        return "blocked"
    if diagnostic_status == "FAIL" or status.upper() == "FAILED":
        return "failed"
    if status.lower() == "completed" and diagnostic_status in {"PASS", "WARN"}:
        return "effective"
    return "blocked"


def _candidate_line(run_id: str, artifact: dict[str, Any] | None) -> str:
    if not isinstance(artifact, dict):
        return f"- {run_id}: No run artifact found"
    status = _artifact_status(artifact)
    diagnostics = _diagnostic_status(artifact)
    factor_ids = artifact.get("factor_ids") or artifact.get("requested_factor_ids") or []
    report_path = artifact.get("report_path", "")
    metrics = artifact.get("metrics")
    details = [
        f"status={status}",
        f"diagnostics={diagnostics}",
        f"factor_ids={factor_ids}",
    ]
    for key in (
        "candidate_type",
        "universe_requested",
        "universe_effective",
        "symbols_source",
        "symbols_count",
        "actual_data_start",
        "actual_data_end",
        "data_freshness",
        "generated_code",
        "static_checks",
        "saved_in_registry",
        "execution_backend",
        "factor_weights",
        "research_only",
        "live_trading_allowed",
    ):
        if key in artifact:
            details.append(f"{key}={artifact[key]}")
    if report_path:
        details.append(f"report_path={report_path}")
    if isinstance(metrics, dict):
        details.append(f"metrics={metrics}")
    warnings = artifact.get("warnings")
    if isinstance(warnings, list) and warnings:
        details.append(f"warnings={warnings}")
    limitations = artifact.get("adapter_limitations")
    if isinstance(limitations, list) and limitations:
        details.append(f"adapter_limitations={limitations}")
    return f"- {run_id}: " + "; ".join(details)


def _artifact_status(artifact: dict[str, Any]) -> str:
    status = artifact.get("status")
    if not status:
        payload = artifact.get("payload")
        status = payload.get("status") if isinstance(payload, dict) else None
    if not status and artifact.get("artifact_type") == "strategy_backtest":
        return "completed"
    return str(status or "unknown")


def _report_evidence_summary(
    run_id: str,
    artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {"run_id": run_id, "status": "missing"}
    data: dict[str, Any] = {
        "run_id": run_id,
        "status": _artifact_status(artifact),
        "diagnostics_status": _diagnostic_status(artifact),
    }
    for key in (
        "strategy_id",
        "factor_ids",
        "requested_factor_ids",
        "execution_backend",
        "composite_method",
        "factor_weights",
        "factor_directions",
        "metrics",
        "data_window",
        "research_only",
        "live_trading_allowed",
        "warnings",
        "adapter_limitations",
    ):
        if key in artifact:
            data[key] = artifact[key]
    config = artifact.get("config")
    if isinstance(config, dict):
        for key in ("symbols", "start_date", "end_date", "universe"):
            if key in config:
                data[key] = config[key]
    return data


def _diagnostic_status(artifact: dict[str, Any]) -> str:
    diagnostics = artifact.get("diagnostics")
    if isinstance(diagnostics, dict):
        return str(diagnostics.get("status", "unknown"))
    return "unknown"


def _report_limitations(artifacts: list[dict[str, Any] | None]) -> list[str]:
    lines: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        for key in ("warnings", "adapter_limitations"):
            value = artifact.get(key)
            if isinstance(value, list):
                lines.extend(f"- {item}" for item in value if str(item).strip())
        diagnostics = artifact.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("status") in {"FAIL", "WARN"}:
            lines.append(f"- Diagnostics status: {diagnostics['status']}")
    return lines


def _report_data_gaps(artifacts: list[dict[str, Any] | None]) -> list[str]:
    lines: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        data_window = artifact.get("data_window")
        if (
            isinstance(data_window, dict)
            and data_window.get("data_freshness") != "covers_requested_end"
        ):
            lines.append(f"- {artifact.get('run_id', 'unknown')}: {data_window}")
    return lines


def _report_diagnostic_gaps(
    run_ids: list[Any],
    artifacts: list[dict[str, Any] | None],
) -> list[str]:
    lines: list[str] = []
    for run_id, artifact in zip(run_ids, artifacts, strict=False):
        if not isinstance(artifact, dict):
            lines.append(f"- {run_id}: run artifact missing, diagnostics unavailable")
            continue
        diagnostics = artifact.get("diagnostics")
        checks = diagnostics.get("checks", []) if isinstance(diagnostics, dict) else []
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, dict) or check.get("status") != "NOT_COMPUTED":
                continue
            lines.append(
                f"- {run_id}: {check.get('name', 'unknown')} NOT_COMPUTED "
                f"({check.get('message', 'no reason captured')})"
            )
    return lines


def _report_next_actions(
    run_ids: list[Any],
    artifacts: list[dict[str, Any] | None],
) -> list[str]:
    actions: list[str] = []
    for run_id, artifact in zip(run_ids, artifacts, strict=False):
        group = _candidate_group(artifact)
        if group == "effective":
            actions.append(
                f"- {run_id}: keep as verifiable candidate; rerun with broader "
                "dates/cost grid before recommendation"
            )
        elif group == "failed":
            actions.append(
                f"- {run_id}: do not recommend as effective; inspect FAIL "
                "diagnostics and revise hypothesis"
            )
        else:
            actions.append(
                f"- {run_id}: unblock missing artifact/data/factor evidence before "
                "comparing performance"
            )
    return actions or ["- No run IDs supplied; run backtests or factor diagnostics first"]


def _factor_registry_root(lake: DataLake) -> Path:
    return lake.root.parent / "factors"


def _strategy_registry() -> StrategyRegistry:
    lake = _get_lake()
    root = (
        lake.root.parent / "strategies"
        if lake is not None
        else PersistencePaths.from_settings(get_settings()).data_root / "strategies"
    )
    if lake is None:
        return StrategyRegistry(root)
    return StrategyRegistry(
        root,
        lock_manager=lake.lock_manager,
        atomic_store=AtomicFileStore(lake.lock_manager),
    )


def _resolve_backtest_universe(
    lake: DataLake,
    input_data: dict[str, Any],
    *,
    strategy_spec: StrategySpec | None,
    saved_strategy: SavedStrategy | None,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    if symbols:
        universe_info: dict[str, Any] = {
            "universe_requested": input_data.get("universe"),
            "universe_effective": input_data.get("universe") or "explicit_symbols",
            "symbols_source": "explicit_symbols",
            "symbols_count": len(symbols),
            "universe_mode": "snapshot",
            "universe_resolve_dates": [],
        }
        return {
            "status": "OK",
            "symbols": symbols,
            "symbols_by_date": None,
            "universe_info": universe_info,
            "resolved_universe": None,
        }

    requested_value = _first_requested_universe_value(
        input_data,
        strategy_spec=strategy_spec,
        saved_strategy=saved_strategy,
    )
    if requested_value is not None and _is_removed_legacy_universe_name(str(requested_value)):
        return {
            "status": "INVALID_REQUEST",
            "payload": {
                "status": "INVALID_REQUEST",
                "reason": "LEGACY_UNIVERSE_NAME_REMOVED",
                "message": (
                    "Legacy thematic universe names have been removed. "
                    "Use universe_id or universe_spec."
                ),
                "universe_requested": str(requested_value),
                "research_only": True,
                "live_trading_allowed": False,
            },
        }

    spec_result = _backtest_universe_spec(
        input_data,
        lake,
        requested_value=requested_value,
        strategy_spec=strategy_spec,
        saved_strategy=saved_strategy,
    )
    if spec_result.get("status") != "OK":
        return {"status": spec_result.get("status", "BLOCKED"), "payload": spec_result}

    spec = spec_result["spec"]
    source = str(spec_result["source"])
    universe_mode = str(input_data.get("universe_mode") or input_data.get("mode") or spec.mode)
    if universe_mode not in {"snapshot", "rolling"}:
        return {
            "status": "INVALID_REQUEST",
            "payload": {
                "status": "INVALID_REQUEST",
                "reason": "UNSUPPORTED_UNIVERSE_MODE",
                "allowed_modes": ["snapshot", "rolling"],
            },
        }
    resolver = UniverseResolver(lake)
    if universe_mode == "rolling":
        resolved = resolver.build(
            spec,
            mode="rolling",
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency=str(
                input_data.get("rebalance_frequency") or spec.rebalance_frequency
            ),
            limit=int(input_data.get("limit", 2000)),
            include_exclusions=bool(input_data.get("include_exclusions", False)),
        )
        if resolved.get("status") != "OK":
            return {"status": str(resolved.get("status")), "payload": resolved}
        metadata = resolved.get("metadata", {})
        empty_dates = [str(item) for item in metadata.get("empty_dates", [])]
        if empty_dates:
            return {
                "status": "BLOCKED",
                "payload": {
                    "status": "BLOCKED",
                    "reason": "ROLLING_UNIVERSE_EMPTY",
                    "empty_dates": empty_dates,
                    "suggested_next_tools": ["inspect_universe", "build_universe", "query_bars"],
                    "universe_resolution": resolved,
                    "universe_mode": "rolling",
                    "rolling_universe_stats": _rolling_universe_stats(metadata),
                },
            }
        rolling_symbols = {
            str(date_key): [str(symbol) for symbol in date_symbols]
            for date_key, date_symbols in resolved.get("rolling_symbols", {}).items()
        }
        symbol_union = _symbols_union(rolling_symbols)
        universe_info = {
            "universe_requested": requested_value,
            "universe_effective": spec.universe_id,
            "universe_id": spec.universe_id,
            "universe_spec_fingerprint": metadata.get("spec_fingerprint"),
            "symbols_source": "universe_rolling",
            "symbols_count": len(symbol_union),
            "universe_mode": "rolling",
            "universe_resolve_dates": metadata.get("resolve_dates", []),
            "rolling_universe_stats": _rolling_universe_stats(metadata),
            "source": source,
        }
        return {
            "status": "OK",
            "symbols": symbol_union,
            "symbols_by_date": rolling_symbols,
            "universe_info": universe_info,
            "resolved_universe": resolved,
        }

    as_of_date = str(input_data.get("as_of_date") or end_date)
    resolved = resolver.build(
        spec,
        mode="snapshot",
        as_of_date=as_of_date,
        limit=int(input_data.get("limit", 2000)),
        include_exclusions=bool(input_data.get("include_exclusions", False)),
    )
    if resolved.get("status") != "OK":
        return {"status": str(resolved.get("status")), "payload": resolved}
    snapshot_symbols = [str(symbol) for symbol in resolved.get("symbols", [])]
    if not snapshot_symbols:
        return {
            "status": "BLOCKED",
            "payload": {
                "status": "BLOCKED",
                "reason": "UNIVERSE_EMPTY",
                "message": "Resolved snapshot universe contains zero symbols.",
                "suggested_next_tools": ["inspect_universe", "build_universe", "query_bars"],
                "universe_resolution": resolved,
                "universe_mode": "snapshot",
            },
        }
    metadata = resolved.get("metadata", {})
    too_small = _broad_universe_too_small(
        source=source,
        spec=spec,
        symbols_count=len(snapshot_symbols),
    )
    if too_small is not None:
        diagnostics = dict(metadata.get("diagnostics") or {})
        diagnostics.update(
            {
                "requested_universe": spec.universe_id,
                "selected_count": len(snapshot_symbols),
                "evidence_threshold": too_small,
                "root_cause_hint": (
                    "Check UniverseResolver as-of snapshot semantics and raw daily coverage."
                ),
            }
        )
        return {
            "status": "BLOCKED",
            "payload": {
                "status": "BLOCKED",
                "reason": "BROAD_UNIVERSE_TOO_SMALL",
                "message": (
                    f"Resolved broad stock universe has {len(snapshot_symbols)} symbols, "
                    f"below minimum evidence threshold {too_small}."
                ),
                "universe_diagnostics": diagnostics,
                "next_repair_action": "repair_universe_resolution_or_market_data_coverage",
                "suggested_next_tools": ["inspect_universe", "query_bars", "plan_tushare_fetch"],
                "universe_resolution": resolved,
                "universe_mode": "snapshot",
                "research_only": True,
                "live_trading_allowed": False,
            },
        }
    universe_info = {
        "universe_requested": requested_value,
        "universe_effective": spec.universe_id,
        "universe_id": spec.universe_id,
        "universe_spec_fingerprint": metadata.get("spec_fingerprint"),
        "symbols_source": "universe_snapshot",
        "symbols_count": len(snapshot_symbols),
        "universe_mode": "snapshot",
        "universe_resolve_dates": [metadata.get("as_of_date")],
        "rolling_universe_stats": None,
        "source": source,
    }
    return {
        "status": "OK",
        "symbols": snapshot_symbols,
        "symbols_by_date": None,
        "universe_info": universe_info,
        "resolved_universe": resolved,
    }


def _first_requested_universe_value(
    input_data: dict[str, Any],
    *,
    strategy_spec: StrategySpec | None,
    saved_strategy: SavedStrategy | None,
) -> Any:
    if input_data.get("universe") is not None:
        return input_data.get("universe")
    if input_data.get("universe_type") is not None:
        return input_data.get("universe_type")
    if strategy_spec is not None and strategy_spec.universe:
        return strategy_spec.universe
    if saved_strategy is not None and saved_strategy.spec.universe:
        return saved_strategy.spec.universe
    return None


def _backtest_universe_spec(
    input_data: dict[str, Any],
    lake: DataLake,
    *,
    requested_value: Any,
    strategy_spec: StrategySpec | None,
    saved_strategy: SavedStrategy | None,
) -> dict[str, Any]:
    if input_data.get("universe_id"):
        registry = UniverseRegistry(registry_root_from_payload(input_data, lake))
        spec = registry.load(str(input_data["universe_id"]))
        if spec is None:
            return {
                "status": "BLOCKED",
                "reason": "UNIVERSE_NOT_FOUND",
                "universe_id": str(input_data["universe_id"]),
                "suggested_next_tools": ["list_universes", "create_universe_spec"],
            }
        return {"status": "OK", "spec": spec, "source": "universe_id"}
    if input_data.get("universe_spec") is not None:
        try:
            return {
                "status": "OK",
                "spec": UniverseSpec.model_validate(input_data["universe_spec"]),
                "source": "universe_spec",
            }
        except Exception as exc:
            return {
                "status": "INVALID_REQUEST",
                "reason": "INVALID_UNIVERSE_SPEC",
                "message": str(exc),
            }
    broad_value = requested_value
    if broad_value is None and saved_strategy is not None and saved_strategy.spec.universe:
        broad_value = saved_strategy.spec.universe
    if broad_value is None and strategy_spec is not None and strategy_spec.universe:
        broad_value = strategy_spec.universe
    if broad_value is None and bool(input_data.get("allow_default_universe")):
        broad_value = "stock_etf"
    if broad_value is None:
        return {
            "status": "BLOCKED",
            "reason": "UNIVERSE_UNSPECIFIED",
            "message": (
                "Backtest would use a default broad universe. Pass symbols, universe_id, "
                "universe_spec, universe_type, or allow_default_universe=true explicitly."
            ),
            "suggested_next_tools": ["create_universe_spec", "build_universe", "query_universe"],
            "universe_requested": None,
            "universe_effective": None,
            "symbols_source": "none",
            "symbols_count": 0,
            "symbols_sample": [],
            "universe_resolution": None,
        }
    try:
        return {
            "status": "OK",
            "spec": broad_universe_spec(
                str(broad_value),
                mode=str(input_data.get("universe_mode") or input_data.get("mode") or "snapshot"),
                rebalance_frequency=str(input_data.get("rebalance_frequency") or "daily"),
            ),
            "source": "broad_universe",
        }
    except ValueError:
        return {
            "status": "INVALID_REQUEST",
            "reason": "UNSUPPORTED_UNIVERSE_REFERENCE",
            "message": "Use universe_id, universe_spec, or broad universe values stock/etf/mixed.",
            "universe_requested": str(broad_value),
        }


def _is_removed_legacy_universe_name(value: str) -> bool:
    normalized = value.strip().lower()
    return "cyclical" in normalized or "顺周期" in value or normalized.startswith("theme:")


def _symbols_union(symbols_by_date: dict[str, list[str]]) -> list[str]:
    symbols: list[str] = []
    for date_symbols in symbols_by_date.values():
        for symbol in date_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _rolling_universe_stats(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_count": int(metadata.get("min_count") or 0),
        "max_count": int(metadata.get("max_count") or 0),
        "mean_count": float(metadata.get("mean_count") or 0.0),
        "empty_dates": [str(item) for item in metadata.get("empty_dates", [])],
        "changed_dates": int(metadata.get("changed_dates") or 0),
    }


def _broad_universe_too_small(
    *,
    source: str,
    spec: UniverseSpec,
    symbols_count: int,
) -> int | None:
    if source != "broad_universe":
        return None
    broad_assets = set(spec.asset_types)
    if not ({"stock"} & broad_assets):
        return None
    if symbols_count >= BROAD_UNIVERSE_MIN_SYMBOLS:
        return None
    return BROAD_UNIVERSE_MIN_SYMBOLS


def _universe_evidence_payload(
    universe_info: dict[str, Any],
    symbols: list[str],
    resolved_universe: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "universe_requested": universe_info.get("universe_requested"),
        "universe_effective": universe_info.get("universe_effective"),
        "universe_mode": universe_info.get("universe_mode", "snapshot"),
        "universe_id": universe_info.get("universe_id"),
        "universe_spec_fingerprint": universe_info.get("universe_spec_fingerprint"),
        "universe_resolve_dates": universe_info.get("universe_resolve_dates", []),
        "symbols_source": universe_info.get("symbols_source", "none"),
        "symbols_count": len(symbols) if symbols else int(universe_info.get("symbols_count") or 0),
        "symbols_sample": symbols[:10],
        "rolling_universe_stats": universe_info.get("rolling_universe_stats"),
        "universe_resolution": resolved_universe,
    }


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


def _backtest_timeout_seconds_for_call(
    input_data: dict[str, Any],
    _context: ToolContext,
) -> int:
    settings = get_settings()
    start = str(input_data.get("start_date", "20200101"))
    end = str(input_data.get("end_date", _today_yyyymmdd()))
    symbols = _requested_symbols(input_data)
    span_days = max(1, _date_span_days(start, end))
    symbol_count = len(symbols) if symbols else 5000
    estimated_rows = span_days * symbol_count
    variable = (
        (estimated_rows + 99_999) // 100_000 * settings.research_tool_timeout_seconds_per_100k_rows
    )
    return int(
        min(
            settings.backtest_tool_max_timeout_seconds,
            max(
                settings.research_tool_base_timeout_seconds,
                settings.research_tool_base_timeout_seconds + variable,
            ),
        )
    )


def _backtest_cost_estimate(config: StrategyBacktestConfig) -> dict[str, Any]:
    estimated_dates = max(1, _date_span_days(config.start_date, config.end_date))
    estimated_symbols = len(config.symbols) if config.symbols else 5000
    estimated_rows = estimated_dates * estimated_symbols
    if estimated_rows < 100_000:
        cost_level = "small"
    elif estimated_rows < 2_000_000:
        cost_level = "medium"
    else:
        cost_level = "large"
    return {
        "estimated_rows": estimated_rows,
        "estimated_dates": estimated_dates,
        "estimated_symbols": estimated_symbols,
        "cost_level": cost_level,
    }


def _backtest_cache_key(
    lake: DataLake,
    *,
    config: StrategyBacktestConfig,
    factor_name: str,
    requested_factor_ids: list[str],
) -> str:
    payload = {
        "schema_version": BACKTEST_CACHE_SCHEMA_VERSION,
        "config": config.model_dump(mode="json"),
        "factor_name": factor_name,
        "requested_factor_ids": requested_factor_ids,
        "data_fingerprint": _data_fingerprint(lake),
        "factor_fingerprint": _factor_fingerprint(lake, requested_factor_ids),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached_backtest(cache_key: str) -> dict[str, Any] | None:
    cache = _cache_var.get()
    return None if cache is None else cache.get("backtest", cache_key)


def _put_cached_backtest(cache_key: str, payload: dict[str, Any]) -> None:
    cache = _cache_var.get()
    if cache is not None:
        cache.put("backtest", cache_key, payload)


def _data_fingerprint(lake: DataLake) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for name in ("tushare/daily", "tushare/fund_daily", "tushare/suspend_d", "tushare/stk_limit"):
        path = lake.dataset_path("raw", name)
        if path.exists():
            stat = path.stat()
            result[name] = (stat.st_mtime_ns, stat.st_size)
    return result


def _factor_fingerprint(lake: DataLake, factor_ids: list[str]) -> dict[str, str]:
    registry = _factor_registry(lake)
    result: dict[str, str] = {}
    for factor_id in factor_ids:
        saved = registry.get_factor(factor_id)
        if saved is None:
            continue
        implementation = str(saved.implementation_ref)
        if implementation.startswith("file:"):
            path = Path(implementation.removeprefix("file:"))
            try:
                stat = path.stat()
            except OSError:
                result[factor_id] = implementation
            else:
                result[factor_id] = f"{implementation}:{stat.st_mtime_ns}:{stat.st_size}"
        else:
            result[factor_id] = f"{implementation}:{saved.version}:{saved.lookback}"
    return result


def _date_span_days(start: str, end: str) -> int:
    start_date = _parse_backtest_date(start)
    end_date = _parse_backtest_date(end)
    return max(1, (end_date - start_date).days + 1)


def _parse_backtest_date(value: str) -> date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(value).date()


def _requested_symbols(input_data: dict[str, Any]) -> list[str]:
    raw_symbols: list[Any] = []
    symbols_value = input_data.get("symbols", [])
    if isinstance(symbols_value, list):
        raw_symbols.extend(symbols_value)
    elif symbols_value:
        raw_symbols.append(symbols_value)
    for alias in ("symbol", "code"):
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
