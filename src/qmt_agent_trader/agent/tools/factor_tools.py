"""Factor tools for factor specs, drafts, saved registry entries, and evaluation."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import FactorSpec, ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    ExecutionStatus,
    RecommendationStatus,
)
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import SHANGHAI_TZ, new_id
from qmt_agent_trader.data.bars import CANONICAL_BAR_COLUMNS
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.context import FactorContext
from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.factors.service import (
    evaluate_factor,
)

_sandbox: CodeSandbox | None = None
_store: ExperimentStore | None = None
_lake: DataLake | None = None
_sandbox_var: ContextVar[CodeSandbox | None] = ContextVar("factor_tool_sandbox", default=None)
_store_var: ContextVar[ExperimentStore | None] = ContextVar("factor_tool_store", default=None)
_lake_var: ContextVar[DataLake | None] = ContextVar("factor_tool_lake", default=None)


def wire(sandbox: CodeSandbox, store: ExperimentStore, lake: DataLake) -> None:
    global _sandbox, _store, _lake
    _sandbox = sandbox
    _store = store
    _lake = lake


def _get_sandbox() -> CodeSandbox | None:
    return _sandbox_var.get() or _sandbox


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


# ── create_factor_spec ──────────────────────────────────────────────────────


def _create_factor_spec(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    hypothesis = input_data.get("hypothesis")
    if isinstance(hypothesis, dict):
        hypothesis = {**hypothesis, **{k: v for k, v in input_data.items() if k != "hypothesis"}}
    else:
        hypothesis = input_data
    name = hypothesis.get("name", "unnamed")
    name = hypothesis.get("factor_name", name)
    required_data = hypothesis.get("required_data") or hypothesis.get("data_sources") or [
        "daily_bars"
    ]
    formula_sketch = hypothesis.get("formula_sketch") or hypothesis.get("factor_description", "")
    lookback = hypothesis.get("lookback", hypothesis.get("expected_holding_period", "20"))
    try:
        lookback = int(str(lookback).replace("d", "").strip())
    except ValueError:
        lookback = 20

    factor_id = new_id("factor")
    spec = FactorSpec(
        factor_id=factor_id,
        name=name,
        version="0.1.0",
        inputs=required_data,
        lookback=lookback,
        formula=formula_sketch,
    )
    return {"factor_spec": spec.model_dump(mode="json")}


create_factor_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_factor_spec",
        description="将自然语言因子假设转成结构化 factor spec。",
        input_schema={
            "type": "object",
            "properties": {
                "hypothesis": {"type": "object"},
                "factor_name": {"type": "string"},
                "factor_description": {"type": "string"},
                "formula_sketch": {"type": "string"},
                "lookback": {"type": "integer"},
                "data_sources": {"type": "array", "items": {"type": "string"}},
                "universe": {"type": "string"},
            },
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_create_factor_spec,
)

# ── generate_factor_code ────────────────────────────────────────────────────


def _generate_factor_code(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = _factor_spec_payload(input_data)
    if not isinstance(spec_data, dict) or not spec_data.get("factor_id"):
        return {
            "status": "INVALID_REQUEST",
            "message": "factor_spec with factor_id is required",
        }
    name = str(spec_data.get("name", "candidate"))
    factor_id = str(spec_data.get("factor_id", new_id("factor")))
    lookback = int(spec_data.get("lookback", 20))
    formula = str(spec_data.get("formula", "return"))
    warnings: list[str] = []
    python_function = str(input_data.get("python_function") or "").strip()
    formula_ast: dict[str, Any] | None
    factor_code: str | None

    if python_function:
        authoring_issues = _factor_python_authoring_issues(python_function)
        if authoring_issues:
            return {
                "status": "STATIC_CHECK_FAILED",
                "factor_id": factor_id,
                "review_required": True,
                "issues": authoring_issues,
                "message": "agent-authored factor function failed static safety checks",
            }
        formula_ast = {"kind": "agent_python_function", "operators": ["python_function"]}
        factor_code = _render_agent_factor_code(spec_data, python_function)
    else:
        formula_ast = _formula_ast_for_supported_formula(name, formula)
        factor_code = _render_factor_code(name, lookback, formula)
        if factor_code is None:
            return {
                "status": "NEEDS_PYTHON_FUNCTION",
                "factor_id": factor_id,
                "review_required": True,
                "message": (
                    "formula sketch is not supported by deterministic generator; "
                    "provide python_function for unrestricted agent-authored factor code"
                ),
                "unsupported_formula": formula,
                "next_required_input": "python_function",
                "suggested_next_tools": ["generate_factor_code"],
                "warnings": ["formula_sketch fallback unsupported; retry with python_function"],
            }
    test_code = _render_factor_test_code(name)
    spec_code = json.dumps(spec_data, ensure_ascii=False, indent=2, default=str)

    sb = _get_sandbox()
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}

    try:
        code_path = sb.write_candidate_file(f"factors/drafts/{factor_id}/factor.py", factor_code)
        tests_path = sb.write_candidate_file(
            f"factors/drafts/{factor_id}/test_factor.py",
            test_code,
        )
        spec_path = sb.write_candidate_file(
            f"factors/drafts/{factor_id}/factor_spec.json",
            spec_code,
        )
        sample_result = _run_factor_sample_test(code_path, spec_data)
        real_schema_result = _run_factor_real_schema_test(spec_data)
        if sample_result["status"] != "PASSED":
            return {
                "status": "SAMPLE_TEST_FAILED",
                "execution_status": ExecutionStatus.OK.value,
                "domain_status": DomainStatus.FAILED.value,
                "evidence_status": EvidenceStatus.INVALID.value,
                "recommendation_status": RecommendationStatus.BLOCKED.value,
                "factor_id": factor_id,
                "code_path": str(code_path),
                "tests_path": str(tests_path),
                "spec_path": str(spec_path),
                "review_required": True,
                "sample_test": sample_result,
                "synthetic_contract_test": sample_result,
                "contract_test_status": sample_result["status"],
                "real_schema_test": real_schema_result,
                "real_schema_status": real_schema_result["status"],
                "next_repair_tool": "generate_factor_code",
                "suggested_repair": _factor_authoring_suggested_repair(),
                "authoring_contract": _factor_authoring_contract(),
                "warnings": [*warnings, "generated factor failed sample execution"],
            }
        payload = {
            "factor_id": factor_id,
            "code_path": str(code_path),
            "tests_path": str(tests_path),
            "spec_path": str(spec_path),
            "status": "generated",
            "execution_status": ExecutionStatus.OK.value,
            "formula_ast": formula_ast,
            "static_check_status": "PASSED",
            "sample_test_status": sample_result["status"],
            "sample_test": sample_result,
            "synthetic_contract_test": sample_result,
            "contract_test_status": sample_result["status"],
            "real_schema_test": real_schema_result,
            "real_schema_status": real_schema_result["status"],
            "review_required": True,
            "research_only": True,
            "live_trading_allowed": False,
            "warnings": warnings,
        }
        if real_schema_result["status"] == "BLOCKED":
            payload.update(
                {
                    "domain_status": DomainStatus.BLOCKED.value,
                    "evidence_status": EvidenceStatus.BLOCKED.value,
                    "recommendation_status": RecommendationStatus.BLOCKED.value,
                    "reason": "MISSING_REAL_DATA_SCHEMA",
                    "missing_columns": real_schema_result["missing_columns"],
                    "warnings": [
                        *warnings,
                        "generated factor code is not runnable evidence on real bars schema",
                    ],
                }
            )
        else:
            payload.update(
                {
                    "domain_status": DomainStatus.WARN.value
                    if real_schema_result["status"] == "UNKNOWN"
                    else DomainStatus.OK.value,
                    "evidence_status": EvidenceStatus.WEAK.value
                    if real_schema_result["status"] == "UNKNOWN"
                    else EvidenceStatus.VALID.value,
                    "recommendation_status": RecommendationStatus.RESEARCH_ONLY.value,
                }
            )
        return payload
    except Exception as exc:
        return {
            "status": "STATIC_CHECK_FAILED",
            "factor_id": factor_id,
            "review_required": True,
            "issues": [str(exc)],
            "warnings": [str(exc)],
        }


generate_factor_code_tool: AgentTool = tool(
    ToolSpec(
        name="generate_factor_code",
        description=(
            "根据 factor spec 生成候选因子代码和测试。主路径支持传入 python_function，"
            "由 Agent 编写 compute_factor(data: pd.DataFrame, context: FactorContext) "
            "函数；工具负责包装、保存、静态检查和样本运行。compute_factor 必须返回 "
            "pd.Series，且 result.index 必须等于 data.index；不要返回 DataFrame，"
            "不要使用 trade_date 或 symbol/trade_date 作为 index。推荐模板："
            "result = pd.Series(factor_values, index=data.index, name='factor_value'); "
            "return pd.to_numeric(result, errors='coerce')。只传 formula_sketch 时会尝试"
            "旧 deterministic fallback，不支持时返回 NEEDS_PYTHON_FUNCTION。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "factor_spec": {"type": "object"},
                "factor_name": {"type": "string"},
                "factor_description": {"type": "string"},
                "python_function": {"type": "string"},
                "tests": {"type": "object"},
            },
            "anyOf": [{"required": ["factor_spec"]}, {"required": ["python_function"]}],
        },
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_factor_code,
)

# ── run_factor_static_checks ────────────────────────────────────────────────


def _run_factor_static_checks(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    code_path_str = input_data.get("code_path", "")
    factor_id = str(input_data.get("factor_id") or "").strip()
    if not code_path_str and not factor_id:
        return {"status": "INVALID_REQUEST", "issues": ["code_path or factor_id is required"]}
    sb = _get_sandbox()
    if sb is None:
        return {"status": "FAILED", "issues": ["sandbox not wired"]}
    code_path, recovered = _resolve_factor_code_path(
        str(code_path_str),
        factor_id=factor_id or None,
        sandbox=sb,
    )
    if not code_path.exists():
        return {"status": "FAILED", "issues": [f"file not found: {code_path_str}"]}
    if not code_path.is_file():
        return {"status": "INVALID_REQUEST", "issues": [f"not a file: {code_path_str}"]}
    code_text = code_path.read_text(encoding="utf-8")
    issues = sb.static_scan_code(code_text)
    issues.extend(_factor_python_authoring_issues(code_text, require_compute_factor=False))
    spec_path = code_path.with_name("factor_spec.json")
    spec_data = _load_factor_spec(str(spec_path), sb) if spec_path.exists() else {}
    semantic_issues = _semantic_issues_for_factor_code(code_text, spec_data)
    issues.extend(semantic_issues)
    return {
        "status": "PASSED" if not issues else "FAILED",
        "issues": issues,
        "semantic_status": "FAILED" if semantic_issues else "PASSED",
        "code_path": str(code_path),
        "path_recovered": recovered,
    }


run_factor_static_checks_tool: AgentTool = tool(
    ToolSpec(
        name="run_factor_static_checks",
        description="检查候选因子是否存在未来函数或危险行为。",
        input_schema={
            "type": "object",
            "properties": {
                "code_path": {"type": "string"},
                "factor_id": {"type": "string"},
            },
        },
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=True,
    ),
    fn=_run_factor_static_checks,
)

# ── save_factor ──────────────────────────────────────────────────────────────


def _save_factor(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    sb = _get_sandbox()
    lake = _get_lake()
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}
    if lake is None:
        return {"status": "error", "message": "data lake not wired"}

    factor_id = str(input_data.get("factor_id") or "").strip()
    code_path_raw = str(input_data.get("code_path") or "").strip()
    spec_path_raw = str(input_data.get("spec_path") or "").strip()
    if not factor_id:
        return {"status": "INVALID_REQUEST", "message": "factor_id is required"}
    if not code_path_raw:
        return {"status": "INVALID_REQUEST", "message": "code_path is required"}

    try:
        code_path = sb.validate_path(code_path_raw)
    except Exception as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}
    if not code_path.exists():
        return {"status": "INVALID_REQUEST", "message": f"file not found: {code_path}"}

    if not spec_path_raw:
        sibling_spec = code_path.with_name("factor_spec.json")
        spec_path_raw = str(sibling_spec) if sibling_spec.exists() else ""
    spec_data = _load_factor_spec(spec_path_raw, sb) if spec_path_raw else {}
    code_text = code_path.read_text(encoding="utf-8")
    issues = sb.static_scan_code(code_text)
    semantic_issues = _semantic_issues_for_factor_code(code_text, spec_data)
    issues.extend(semantic_issues)
    if issues:
        return {
            "status": "FAILED",
            "issues": issues,
            "semantic_status": "FAILED" if semantic_issues else "PASSED",
        }
    if spec_data.get("factor_id") and spec_data["factor_id"] != factor_id:
        return {
            "status": "INVALID_REQUEST",
            "message": "factor_id does not match factor_spec",
        }
    registry = FactorRegistry(_factor_registry_root(lake))
    try:
        saved = registry.save_factor(
            factor_id=factor_id,
            name=str(spec_data.get("name") or factor_id),
            version=str(spec_data.get("version") or "0.1.0"),
            implementation_ref=f"file:{code_path}",
            required_columns=_required_columns_for_spec(spec_data),
            lookback=int(spec_data.get("lookback") or 20),
            params={"lookback": int(spec_data.get("lookback") or 20)},
            created_by=str(input_data.get("created_by") or "agent"),
        )
    except ValueError as exc:
        return {
            "status": "DUPLICATE_FACTOR_NAME",
            "message": str(exc),
            "factor_id": factor_id,
            "existing_factors": _factor_matches(registry, str(spec_data.get("name") or factor_id)),
        }
    return {
        "status": saved.status,
        "factor_id": saved.factor_id,
        "name": saved.name,
        "registry_path": str(registry.registry_path),
        "implementation_ref": saved.implementation_ref,
    }


save_factor_tool: AgentTool = tool(
    ToolSpec(
        name="save_factor",
        description=(
            "将通过静态检查的候选因子保存到统一 Factor Registry。保存后才可评估或回测。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "factor_id": {"type": "string"},
                "code_path": {"type": "string"},
                "spec_path": {"type": "string"},
                "created_by": {"type": "string"},
            },
            "required": ["factor_id", "code_path"],
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_save_factor,
)

# ── list_saved_factors ───────────────────────────────────────────────────────


def _list_saved_factors(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_IMPLEMENTED", "message": "data lake not wired"}

    registry = FactorRegistry(_factor_registry_root(lake))
    query = str(input_data.get("query") or "").strip()
    include_builtins = bool(input_data.get("include_builtins", True))
    exact = bool(input_data.get("exact", False))
    include_usage_hints = bool(input_data.get("include_usage_hints", False))
    implementation_type = str(input_data.get("implementation_type", "any")).strip().lower()
    required_columns_any = [
        str(item)
        for item in input_data.get("required_columns_any", [])
        if str(item).strip()
    ]
    limit = max(1, int(input_data.get("limit") or 50))
    factors = registry.find_factors(query or None, include_builtins=include_builtins)
    factors = [
        item
        for item in factors
        if _factor_payload_matches_filters(
            item,
            implementation_type=implementation_type,
            required_columns_any=required_columns_any,
        )
    ][:limit]
    payloads = [
        _saved_factor_payload(item, include_usage_hints=include_usage_hints)
        for item in factors
    ]
    exact_matches = [
        item
        for item in payloads
        if query
        and (
            str(item.get("factor_id", "")).lower() == query.lower()
            or str(item.get("name", "")).lower() == query.lower()
        )
    ]
    candidates = exact_matches if exact else payloads
    duplicates = {
        name: [
            _saved_factor_payload(item, include_usage_hints=include_usage_hints)
            for item in items
        ]
        for name, items in registry.duplicate_names().items()
        if include_builtins or not any(
            item.implementation_ref.startswith("builtin:") for item in items
        )
    }
    return {
        "status": "ok",
        "query": query or None,
        "include_builtins": include_builtins,
        "exact": exact,
        "count": len(payloads),
        "factors": payloads,
        "exact_matches": exact_matches,
        "candidates": candidates,
        "duplicate_names": duplicates,
    }


list_saved_factors_tool: AgentTool = tool(
    ToolSpec(
        name="list_saved_factors",
        description=(
            "查询统一 Factor Registry 中已保存的因子、状态和重复名称。"
            "在创建、评估或回测新因子前先用它确认已有 factor_id/name。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "include_builtins": {"type": "boolean"},
                "exact": {"type": "boolean"},
                "required_columns_any": {"type": "array", "items": {"type": "string"}},
                "implementation_type": {
                    "type": "string",
                    "enum": ["builtin", "file", "any"],
                },
                "limit": {"type": "integer", "minimum": 1},
                "include_usage_hints": {"type": "boolean"},
            },
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_list_saved_factors,
)

# ── evaluate_factor_candidate ────────────────────────────────────────────────


def _evaluate_factor_candidate(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    factor_id = input_data.get("factor_id", "")
    if not str(factor_id).strip():
        return {"status": "INVALID_REQUEST", "message": "factor_id is required"}
    start = input_data.get("start_date", "20200101")
    end = input_data.get("end_date", _today_yyyymmdd())
    symbols = _requested_symbols(input_data)
    if not symbols and _date_span_days(str(start), str(end)) > 366:
        return {
            "status": "BLOCKED",
            "reason": "UNBOUNDED_FACTOR_EVALUATION",
            "message": (
                "factor evaluation over more than 366 days requires explicit symbols; "
                "call query_universe/query_bars first or narrow the date window"
            ),
            "start_date": start,
            "end_date": end,
            "missing_inputs": ["symbols"],
            "next_repair_tool": "query_universe",
        }

    lake = _get_lake()
    if lake is None:
        return {"status": "NOT_IMPLEMENTED", "message": "data lake not wired"}

    factor_name = str(factor_id).strip()
    registry_root = _factor_registry_root(lake)
    registry = FactorRegistry(registry_root)
    saved = registry.get_factor(factor_name)
    if saved is None:
        return {
            "status": "FACTOR_NOT_FOUND",
            "message": (
                f"factor '{factor_name}' is not an exact saved factor_id/name. "
                "Call list_saved_factors and use an exact factor_id."
            ),
            "candidates": _factor_matches(registry, factor_name),
        }
    factor_name = saved.factor_id

    window_days = int(input_data.get("window_days", 63))
    step_days = int(input_data.get("step_days", 63))
    quantile = float(input_data.get("quantile", 0.20))

    # ── Dedup: check cache first ──
    from qmt_agent_trader.agent.tools.cache import (
        get_cached_validation,
        put_cached_validation,
    )
    cache_factor_name = json.dumps(
        {
            "factor_id": factor_name,
            "start": start,
            "end": end,
            "symbols": sorted(symbols),
            "registry_version": _factor_cache_version(saved),
            "window_days": window_days,
            "step_days": step_days,
            "quantile": quantile,
        },
        sort_keys=True,
    )
    cached = get_cached_validation(cache_factor_name, start, end)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    try:
        bundle = evaluate_factor(
            lake,
            name=factor_name,
            start=start,
            end=end,
            registry_root=str(registry_root),
            symbols=symbols or None,
            window_days=window_days,
            step_days=step_days,
            quantile=quantile,
        )
        result = bundle.validation.as_dict()
        result["walk_forward"] = bundle.walk_forward.as_dict()["walk_forward"]
        result["quantile_returns"] = bundle.quantile_returns
        sb = _get_sandbox()
        result["report_path"] = str(
            sb.generated_root / "reports" / f"factor_{factor_id}.json" if sb else ""
        )
        result["cache_hit"] = False
        put_cached_validation(cache_factor_name, start, end, result)
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


evaluate_factor_candidate_tool: AgentTool = tool(
    ToolSpec(
        name="evaluate_factor_candidate",
        description="计算并评估候选因子。",
        input_schema={
            "type": "object",
            "properties": {
                "factor_id": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "symbol": {"type": "string"},
                "code": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["factor_id"],
        },
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=False,
        timeout_seconds=120,
    ),
    fn=_evaluate_factor_candidate,
    timeout_seconds_for_call=lambda input_data, context: _factor_eval_timeout_seconds_for_call(
        input_data,
        context,
    ),
)


def build_factor_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    definitions: list[
        tuple[AgentTool, Callable[[dict[str, Any], ToolContext], dict[str, Any]]]
    ] = [
        (create_factor_spec_tool, _create_factor_spec),
        (generate_factor_code_tool, _generate_factor_code),
        (run_factor_static_checks_tool, _run_factor_static_checks),
        (save_factor_tool, _save_factor),
        (list_saved_factors_tool, _list_saved_factors),
        (evaluate_factor_candidate_tool, _evaluate_factor_candidate),
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


def _factor_spec_payload(input_data: dict[str, Any]) -> dict[str, Any]:
    raw = input_data.get("factor_spec")
    if isinstance(raw, dict):
        payload = dict(raw)
    else:
        payload = {}
    payload.setdefault("factor_id", new_id("factor"))
    payload.setdefault(
        "name",
        input_data.get("factor_name") or payload["factor_id"],
    )
    payload.setdefault(
        "description",
        input_data.get("factor_description") or payload.get("formula") or "",
    )
    payload.setdefault("formula", payload.get("description", "agent-authored python function"))
    payload.setdefault("version", "0.1.0")
    payload.setdefault("lookback", 20)
    if "inputs" not in payload and "required_columns" in payload:
        payload["inputs"] = list(payload["required_columns"])
    if "required_columns" not in payload:
        payload["required_columns"] = list(_required_columns_for_spec(payload))
    return payload


def _factor_registry_root(lake: DataLake) -> Path:
    return lake.root.parent / "factors"


def _saved_factor_payload(
    saved: Any,
    *,
    include_usage_hints: bool = False,
) -> dict[str, Any]:
    implementation_type = (
        "builtin"
        if str(saved.implementation_ref).startswith("builtin:")
        else "file"
    )
    payload = {
        "factor_id": saved.factor_id,
        "name": saved.name,
        "status": saved.status,
        "version": saved.version,
        "created_by": saved.created_by,
        "created_at": saved.created_at,
        "implementation_type": implementation_type,
        "required_columns": list(saved.required_columns),
        "lookback": saved.lookback,
    }
    if include_usage_hints:
        payload["strategy_leg_example"] = {
            "factor_id": saved.factor_id,
            "weight": 1.0,
            "ascending": _default_factor_leg_ascending(saved.factor_id),
        }
        payload["usage_hint"] = (
            "Use this exact factor_id in strategy factor leg objects; "
            "factor_name is not a valid strategy leg field."
        )
    return payload


def _factor_payload_matches_filters(
    saved: Any,
    *,
    implementation_type: str,
    required_columns_any: list[str],
) -> bool:
    actual_type = (
        "builtin"
        if str(saved.implementation_ref).startswith("builtin:")
        else "file"
    )
    if implementation_type and implementation_type != "any" and implementation_type != actual_type:
        return False
    if required_columns_any:
        required = {str(column) for column in saved.required_columns}
        wanted = {str(column) for column in required_columns_any}
        if required.isdisjoint(wanted):
            return False
    return True


def _default_factor_leg_ascending(factor_id: str) -> bool:
    lowered = str(factor_id).lower()
    return any(
        marker in lowered
        for marker in (
            "volatility",
            "turnover",
            "drawdown",
            "risk",
            "reversal",
        )
    )


def _factor_cache_version(saved: Any) -> str:
    implementation = str(getattr(saved, "implementation_ref", ""))
    if implementation.startswith("file:"):
        path = Path(implementation.removeprefix("file:"))
        try:
            stat = path.stat()
        except OSError:
            return implementation
        return f"{implementation}:{stat.st_mtime_ns}:{stat.st_size}"
    return f"{implementation}:{getattr(saved, 'version', '')}:{getattr(saved, 'lookback', '')}"


def _factor_eval_timeout_seconds_for_call(
    input_data: dict[str, Any],
    _context: ToolContext,
) -> int:
    settings = get_settings()
    symbols = _requested_symbols(input_data)
    span_days = max(
        1,
        _date_span_days(
            str(input_data.get("start_date", "20200101")),
            str(input_data.get("end_date", _today_yyyymmdd())),
        ),
    )
    symbol_count = len(symbols) if symbols else 5000
    estimated_rows = span_days * symbol_count
    variable = (
        (estimated_rows + 99_999)
        // 100_000
        * settings.research_tool_timeout_seconds_per_100k_rows
    )
    return int(
        min(
            settings.factor_eval_tool_max_timeout_seconds,
            max(
                settings.research_tool_base_timeout_seconds,
                settings.research_tool_base_timeout_seconds + variable,
            ),
        )
    )


def _factor_matches(registry: FactorRegistry, query: str) -> list[dict[str, Any]]:
    return [
        _saved_factor_payload(item)
        for item in registry.find_factors(query, include_builtins=True)
    ]


def _resolve_factor_code_path(
    code_path_raw: str,
    *,
    factor_id: str | None,
    sandbox: CodeSandbox,
) -> tuple[Path, bool]:
    if factor_id:
        candidate = sandbox.generated_root / "factors" / "drafts" / factor_id / "factor.py"
        if candidate.exists():
            return candidate, not code_path_raw or str(candidate) != code_path_raw
    code_path = Path(code_path_raw)
    if code_path.exists():
        return code_path, False
    match = re.search(r"(factor_[A-Za-z0-9_]+)", code_path_raw)
    if match:
        candidate = sandbox.generated_root / "factors" / "drafts" / match.group(1) / "factor.py"
        if candidate.exists():
            return candidate, True
    return code_path, False


def _load_factor_spec(spec_path_raw: str, sb: CodeSandbox) -> dict[str, Any]:
    try:
        spec_path = sb.validate_path(spec_path_raw)
    except Exception:
        return {}
    if not spec_path.exists():
        return {}
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _required_columns_for_spec(spec_data: dict[str, Any]) -> tuple[str, ...]:
    explicit = spec_data.get("required_columns") or spec_data.get("inputs")
    formula = json.dumps(spec_data, ensure_ascii=False).lower()
    if isinstance(explicit, list):
        columns = ["symbol", "trade_date"]
        data_sources = {
            "daily_bars",
            "tushare_daily",
            "tushare/daily",
            "bars",
            "fundamentals",
            "macro",
            "macro_series",
        }
        for item in explicit:
            text = str(item)
            if text in data_sources:
                continue
            if text and text not in columns:
                columns.append(text)
        if "close" not in columns:
            columns.append("close")
        _append_semantic_required_columns(columns, formula)
        return tuple(columns)
    columns = ["symbol", "trade_date", "close"]
    _append_semantic_required_columns(columns, formula)
    return tuple(columns)


def _append_semantic_required_columns(columns: list[str], formula: str) -> None:
    for candidate in (
        "open",
        "high",
        "low",
        "volume",
        "amount",
        "turnover",
        "industry",
        "macro_cycle_score",
    ):
        if candidate in formula and candidate not in columns:
            columns.append(candidate)
    if "换手" in formula and "turnover" not in columns:
        columns.append("turnover")
    if "行业" in formula and "industry" not in columns:
        columns.append("industry")
    if "宏观" in formula and "macro_cycle_score" not in columns:
        columns.append("macro_cycle_score")


def _render_agent_factor_code(spec_data: dict[str, Any], python_function: str) -> str:
    spec_literal = json.dumps(spec_data, ensure_ascii=False, indent=2, default=str)
    return f'''"""Agent-generated factor. REVIEW_REQUIRED before promotion."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from qmt_agent_trader.factors.context import FactorContext


FACTOR_SPEC = {spec_literal}


{python_function}


def compute(bars: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    """Compatibility wrapper for FactorRegistry."""
    params = params or {{}}
    context = FactorContext(
        factor_id=str(FACTOR_SPEC.get("factor_id", "agent_factor")),
        lookback=int(params.get("lookback", FACTOR_SPEC.get("lookback", 20))),
        params=params,
        as_of_date=params.get("as_of_date"),
        research_only=True,
    )
    result = compute_factor(bars.copy(deep=True), context)
    if not isinstance(result, pd.Series):
        raise ValueError("compute_factor must return pandas Series")
    return pd.to_numeric(result.reindex(bars.index), errors="coerce")
'''


_FACTOR_FORBIDDEN_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "urllib",
    "multiprocessing",
    "threading",
    "pickle",
    "joblib",
}
_FACTOR_FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "input", "__import__"}
_FACTOR_LIVE_TRADING_NAMES = {
    "broker",
    "gateway",
    "xtquant",
    "submit_order",
    "submit_live_order",
    "approve_strategy",
}


def _factor_python_authoring_issues(
    code: str,
    *,
    require_compute_factor: bool = True,
) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]

    issues: list[str] = []
    has_compute_factor = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_factor":
            has_compute_factor = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FACTOR_FORBIDDEN_IMPORT_ROOTS:
                    issues.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in _FACTOR_FORBIDDEN_IMPORT_ROOTS:
                issues.append(f"forbidden import: {module}")
            if module.startswith("qmt_agent_trader.broker") or module.startswith(
                "qmt_agent_trader.gateway"
            ):
                issues.append(f"forbidden live trading import: {module}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in _FACTOR_FORBIDDEN_CALLS:
                issues.append(f"forbidden call: {call_name}")
            if call_name.endswith(".shift") and _has_negative_numeric_arg(node):
                issues.append("future data access: negative shift")
            if call_name.endswith(".pct_change") and _has_negative_numeric_arg(node):
                issues.append("future data access: negative pct_change")
            if call_name.endswith(".rolling") and _has_true_keyword(node, "center"):
                issues.append("future data risk: rolling(center=True)")
        elif isinstance(node, ast.Attribute) and node.attr == "environ":
            issues.append("forbidden environment access")

    lowered = code.lower()
    for token in _FACTOR_LIVE_TRADING_NAMES:
        if token in lowered:
            issues.append(f"forbidden live trading reference: {token}")
    if require_compute_factor and not has_compute_factor:
        issues.append("missing required function: compute_factor")
    return sorted(set(issues))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _has_negative_numeric_arg(node: ast.Call) -> bool:
    for arg in node.args:
        if _is_negative_numeric(arg):
            return True
    for keyword in node.keywords:
        if _is_negative_numeric(keyword.value):
            return True
    return False


def _is_negative_numeric(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int | float)
    )


def _has_true_keyword(node: ast.Call, name: str) -> bool:
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is True
    return False


def _run_factor_sample_test(code_path: Path, spec_data: dict[str, Any]) -> dict[str, Any]:
    try:
        module = _load_generated_factor_module(code_path)
        compute = getattr(module, "compute", None)
        compute_factor = getattr(module, "compute_factor", None)
        if not callable(compute) or not callable(compute_factor):
            return {"status": "FAILED", "issues": ["compute and compute_factor must be callable"]}
        sample = _sample_factor_data(spec_data)
        direct_input = sample.copy(deep=True)
        direct_before = direct_input.copy(deep=True)
        context = FactorContext(
            factor_id=str(spec_data.get("factor_id", "agent_factor")),
            lookback=int(spec_data.get("lookback", 20)),
            params={},
            research_only=True,
        )
        direct_result = compute_factor(direct_input, context)
        direct_issues = _factor_sample_issues(
            direct_result,
            direct_input,
            direct_before,
            source="compute_factor",
        )
        if direct_issues:
            return {
                "status": "FAILED",
                "issues": direct_issues,
                "rows": len(sample),
                "contract_test_only": True,
                "non_null": int(pd.to_numeric(direct_result, errors="coerce").notna().sum())
                if isinstance(direct_result, pd.Series)
                else 0,
            }
        before = sample.copy(deep=True)
        try:
            result = compute(sample)
        except ValueError as exc:
            return {
                "status": "FAILED",
                "issues": [_factor_sample_exception_hint(str(exc))],
                "rows": len(sample),
                "contract_test_only": True,
                "non_null": 0,
            }
        issues = _factor_sample_issues(result, sample, before, source="compute")
        return {
            "status": "PASSED" if not issues else "FAILED",
            "issues": issues,
            "rows": len(sample),
            "contract_test_only": True,
            "non_null": int(pd.to_numeric(result, errors="coerce").notna().sum())
            if isinstance(result, pd.Series)
            else 0,
        }
    except Exception as exc:
        return {"status": "FAILED", "issues": [str(exc)]}


def _load_generated_factor_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"agent_factor_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load generated factor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_factor_data(spec_data: dict[str, Any]) -> pd.DataFrame:
    required = _required_columns_for_spec(spec_data)
    rows: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(["000001.SZ", "000002.SZ"]):
        for offset in range(80):
            row: dict[str, Any] = {
                "symbol": symbol,
                "trade_date": f"202401{offset + 1:02d}",
                "open": 10.0 + symbol_index + offset * 0.1,
                "high": 10.5 + symbol_index + offset * 0.1,
                "low": 9.5 + symbol_index + offset * 0.1,
                "close": 10.2 + symbol_index + offset * (0.2 if symbol_index == 0 else 0.05),
                "volume": 1000 + offset * 10 + symbol_index,
                "vol": 1000 + offset * 10 + symbol_index,
                "amount": 10_000 + offset * 100 + symbol_index,
            }
            for column in required:
                row.setdefault(column, 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _run_factor_real_schema_test(spec_data: dict[str, Any]) -> dict[str, Any]:
    required = set(_required_columns_for_spec(spec_data))
    real_columns = set(CANONICAL_BAR_COLUMNS)
    missing = sorted(required.difference(real_columns))
    warnings: list[str] = []
    status = "PASSED"
    if missing:
        status = "BLOCKED"
    elif "turnover" in required:
        status = "UNKNOWN"
        warnings.append("TURNOVER_REQUIRES_RUNTIME_COLUMN_QUALITY")
    return {
        "status": status,
        "required_columns": sorted(required),
        "real_schema_columns": sorted(real_columns),
        "missing_columns": missing,
        "reason": "MISSING_REAL_DATA_SCHEMA" if missing else None,
        "warnings": warnings,
        "real_schema_test_only": True,
    }


def _factor_sample_issues(
    result: Any,
    sample: pd.DataFrame,
    before: pd.DataFrame,
    *,
    source: str,
) -> list[str]:
    issues: list[str] = []
    if not sample.equals(before):
        issues.append("compute_factor mutated input DataFrame")
    if isinstance(result, pd.DataFrame):
        return [
            *issues,
            (
                "compute_factor returned DataFrame; return pandas Series with "
                "pd.Series(..., index=data.index, name='factor_value') instead"
            ),
        ]
    if not isinstance(result, pd.Series):
        return [*issues, f"{source} must return pandas Series"]
    if not result.index.equals(sample.index):
        issues.append(_factor_index_mismatch_hint(result, sample))
    numeric = pd.to_numeric(result, errors="coerce")
    if numeric.notna().sum() == 0:
        issues.append("result must not be entirely null")
    if np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).any():
        issues.append("result must not contain infinite values")
    return issues


def _factor_index_mismatch_hint(result: pd.Series, sample: pd.DataFrame) -> str:
    result_index = result.index
    input_index = sample.index
    details = [
        "result index must match input index",
        f"input_index_type={type(input_index).__name__}",
        f"result_index_type={type(result_index).__name__}",
        f"input_len={len(input_index)}",
        f"result_len={len(result_index)}",
        f"result_has_duplicates={bool(result_index.has_duplicates)}",
    ]
    if result_index.has_duplicates:
        details.append("duplicate labels detected")
    if len(result_index) == len(sample) and result_index.equals(pd.Index(sample["trade_date"])):
        details.append("do not use trade_date as the factor index")
    if isinstance(result_index, pd.MultiIndex):
        details.append("do not use symbol/trade_date MultiIndex as the factor index")
    return "; ".join(details)


def _factor_sample_exception_hint(message: str) -> str:
    if "duplicate labels" in message:
        return (
            f"{message}; duplicate labels usually mean compute_factor used trade_date "
            "or symbol/trade_date as the index. Return pd.Series(..., index=data.index, "
            "name='factor_value') instead."
        )
    return message


def _factor_authoring_contract() -> dict[str, Any]:
    return {
        "function": "compute_factor(data: pd.DataFrame, context: FactorContext)",
        "return_type": "pd.Series",
        "index": "result.index must equal data.index",
        "forbidden_outputs": ["DataFrame", "trade_date index", "symbol/trade_date MultiIndex"],
        "mutates_input": False,
    }


def _factor_authoring_suggested_repair() -> str:
    return (
        "Return a row-aligned numeric Series: "
        "result = pd.Series(factor_values, index=data.index, name='factor_value'); "
        "return pd.to_numeric(result, errors='coerce'). Do not return a DataFrame "
        "and do not index by trade_date or symbol/trade_date."
    )


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


def _render_factor_code(name: str, lookback: int, formula: str) -> str | None:
    safe_name = name.replace(" ", "_").replace("-", "_").lower()
    formula_lower = formula.lower()
    body = _factor_compute_body(safe_name, formula_lower, lookback)
    if body is None:
        return None
    return f'''"""Candidate factor: {name}.
Auto-generated by Agent. REVIEW_REQUIRED before promotion.
"""

from typing import Any

import pandas as pd
import numpy as np

from qmt_agent_trader.factors.context import FactorContext


def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """{formula}

    Lookback: {lookback} days.
    """
    bars = data
    if bars.empty:
        return pd.Series(dtype="float64")
    lookback = int(context.lookback)
{body}


def compute(bars: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    context = FactorContext(
        factor_id="{safe_name}",
        lookback=int((params or {{}}).get("lookback", {lookback})),
        params=params or {{}},
        as_of_date=(params or {{}}).get("as_of_date"),
        research_only=True,
    )
    result = compute_factor(bars.copy(deep=True), context)
    return pd.to_numeric(result.reindex(bars.index), errors="coerce")


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or float(std) == 0.0:
        return values * 0.0
    return (values - values.mean()) / std
'''


def _factor_compute_body(name: str, formula: str, lookback: int) -> str | None:
    formula_ast = _formula_ast_for_supported_formula(name, formula)
    if formula_ast and formula_ast["kind"] == "low_vol_inverse":
        return '''    previous_close = bars.groupby("symbol")["close"].shift(1)
    log_return = np.log(
        pd.to_numeric(bars["close"], errors="coerce")
        / pd.to_numeric(previous_close, errors="coerce")
    )
    volatility = log_return.groupby(bars["symbol"]).transform(
        lambda item: item.rolling(lookback).std()
    )
    return 1.0 / (volatility + 1e-9)'''
    if formula_ast and formula_ast["kind"] == "negative_rolling_std":
        return '''    close = pd.to_numeric(bars["close"], errors="coerce")
    rolling_std = close.groupby(bars["symbol"]).transform(
        lambda item: item.rolling(lookback).std()
    )
    return -rolling_std'''
    if formula_ast and formula_ast["kind"] == "price_position":
        return '''    close = pd.to_numeric(bars["close"], errors="coerce")
    rolling_min = close.groupby(bars["symbol"]).transform(
        lambda item: item.rolling(lookback).min()
    )
    rolling_max = close.groupby(bars["symbol"]).transform(
        lambda item: item.rolling(lookback).max()
    )
    price_position = (close - rolling_min) / (rolling_max - rolling_min + 1e-9)
    return price_position'''
    if _is_low_vol_low_turnover_formula(name, formula):
        return '''    if "turnover" not in bars.columns:
        raise ValueError("low-volatility + low-turnover factor requires turnover column")
    returns = bars.groupby("symbol")["close"].pct_change()
    volatility = returns.groupby(bars["symbol"]).transform(
        lambda item: item.rolling(lookback).std()
    )
    rolling_turnover = pd.to_numeric(bars["turnover"], errors="coerce").groupby(
        bars["symbol"]
    ).transform(lambda item: item.rolling(lookback).mean())
    vol_component = -volatility
    turnover_component = -rolling_turnover
    if "trade_date" in bars.columns:
        vol_z = vol_component.groupby(bars["trade_date"]).transform(_zscore)
        turnover_z = turnover_component.groupby(bars["trade_date"]).transform(_zscore)
    else:
        vol_z = _zscore(vol_component)
        turnover_z = _zscore(turnover_component)
    composite = (vol_z.fillna(0.0) + turnover_z.fillna(0.0)) / 2.0
    return composite'''
    if _is_sector_neutral_low_vol_formula(name, formula):
        return '''    if "industry" not in bars.columns:
        raise ValueError("sector-neutral low-volatility factor requires industry column")
    returns = bars.groupby("symbol")["close"].pct_change()
    volatility = returns.groupby(bars["symbol"]).transform(
        lambda item: item.rolling(lookback).std()
    )
    raw_score = -volatility
    if "trade_date" in bars.columns:
        neutralized = raw_score - raw_score.groupby(
            [bars["trade_date"], bars["industry"]]
        ).transform("mean")
    else:
        neutralized = raw_score - raw_score.groupby(bars["industry"]).transform("mean")
    return neutralized'''
    if _is_low_vol_formula(name, formula):
        return '''    returns = bars.groupby("symbol")["close"].pct_change()
    volatility = returns.groupby(bars["symbol"]).transform(
        lambda item: item.rolling(lookback).std()
    )
    return -volatility'''
    if _is_low_turnover_formula(name, formula):
        return '''    if "turnover" not in bars.columns:
        raise ValueError("low-turnover factor requires turnover column")
    rolling_turnover = pd.to_numeric(bars["turnover"], errors="coerce").groupby(
        bars["symbol"]
    ).transform(lambda item: item.rolling(lookback).mean())
    return -rolling_turnover'''
    if _is_macro_timed_momentum_formula(name, formula):
        return '''    if "macro_cycle_score" not in bars.columns:
        raise ValueError("macro-timed momentum factor requires macro_cycle_score column")
    momentum = bars.groupby("symbol")["close"].pct_change(lookback)
    gate = pd.to_numeric(bars["macro_cycle_score"], errors="coerce").clip(lower=0.0)
    return momentum * gate'''
    if _is_rsi_formula(name, formula):
        return '''    delta = bars.groupby("symbol")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(bars["symbol"]).transform(lambda item: item.rolling(14).mean())
    avg_loss = loss.groupby(bars["symbol"]).transform(lambda item: item.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)'''
    if "volume_breakout" in name or "volume_ratio" in formula:
        return '''    volume_ma = bars.groupby("symbol")["volume"].transform(
        lambda item: item.rolling(5).mean()
    )
    return bars["volume"] / volume_ma'''
    if "intraday_range" in name or "(high-low)" in formula or "high-low" in formula:
        return '''    daily_range = (bars["high"] - bars["low"]) / bars["close"]
    return daily_range.groupby(bars["symbol"]).transform(lambda item: item.rolling(5).mean())'''
    if "ma_cross" in name or ("ma5" in formula and "ma20" in formula):
        return '''    ma5 = bars.groupby("symbol")["close"].transform(
        lambda item: item.rolling(5).mean()
    )
    ma20 = bars.groupby("symbol")["close"].transform(
        lambda item: item.rolling(20).mean()
    )
    return ma5 / ma20 - 1'''
    if "trend_persistence" in name or "ma60" in formula:
        return '''    moving_average = bars.groupby("symbol")["close"].transform(
        lambda item: item.rolling(lookback).mean()
    )
    return bars["close"] / moving_average - 1'''
    if _is_momentum_formula(name, formula):
        return '    return bars.groupby("symbol")["close"].pct_change(lookback)'
    return None


def _is_low_vol_low_turnover_formula(name: str, formula: str) -> bool:
    text = f"{name} {formula}"
    return _has_low_vol(text) and _has_turnover(text)


def _formula_ast_for_supported_formula(name: str, formula: str) -> dict[str, Any] | None:
    text = f"{name} {formula}".lower().replace(" ", "")
    if "std(log_return" in text and ("1.0/" in text or "1/" in text):
        return {
            "kind": "low_vol_inverse",
            "operators": ["log_return", "rolling_std", "inverse"],
        }
    if text.startswith("low_vol") and "std(close" in text:
        return {
            "kind": "negative_rolling_std",
            "operators": ["rolling_std", "negate"],
        }
    if "price_position" in text and "min(close" in text and "max(close" in text:
        return {
            "kind": "price_position",
            "operators": ["rolling_min", "rolling_max", "arithmetic_ratio"],
        }
    return None


def _is_sector_neutral_low_vol_formula(name: str, formula: str) -> bool:
    text = f"{name} {formula}"
    return _has_low_vol(text) and ("sector neutral" in text or "industry" in text or "行业" in text)


def _is_low_vol_formula(name: str, formula: str) -> bool:
    return _has_low_vol(f"{name} {formula}")


def _is_low_turnover_formula(name: str, formula: str) -> bool:
    text = f"{name} {formula}"
    return _has_turnover(text) and ("low" in text or "低" in text)


def _is_macro_timed_momentum_formula(name: str, formula: str) -> bool:
    text = f"{name} {formula}"
    has_macro_gate = "macro" in text or "宏观" in text or "macro_cycle_score" in text
    return has_macro_gate and _is_momentum_formula(name, formula)


def _is_momentum_formula(name: str, formula: str) -> bool:
    text = f"{name} {formula}"
    return "momentum" in text or "pct_change" in text or "动量" in text


def _has_low_vol(text: str) -> bool:
    return (
        "low volatility" in text
        or "low-volatility" in text
        or "low_vol" in text
        or ("volatility" in text and "low" in text)
        or "低波" in text
    )


def _has_turnover(text: str) -> bool:
    return "turnover" in text or "换手" in text


def _is_rsi_formula(name: str, formula: str) -> bool:
    tokens = {
        token
        for token in name.replace("-", "_").split("_")
        if token
    }
    return "rsi" in tokens or "relative strength" in formula or "rs =" in formula


def _semantic_issues_for_factor_code(code: str, spec_data: dict[str, Any]) -> list[str]:
    if not spec_data:
        return []
    formula = str(spec_data.get("formula") or "").lower()
    name = str(spec_data.get("name") or "").lower()
    text = f"{name} {formula}"
    code_lower = code.lower()
    issues: list[str] = []
    if _has_low_vol(text):
        if ".std(" not in code_lower:
            issues.append("semantic mismatch: low volatility formula must compute rolling std")
        if 'pct_change(lookback)' in code_lower and ".std(" not in code_lower:
            issues.append("semantic mismatch: low volatility formula generated momentum fallback")
    if _has_turnover(text) and '"turnover"' not in code_lower and "'turnover'" not in code_lower:
        issues.append("semantic mismatch: turnover formula must reference turnover column")
    has_sector_neutral = "sector neutral" in text or "industry" in text or "行业" in text
    if has_sector_neutral and "industry" not in code_lower:
        issues.append("semantic mismatch: sector-neutral formula must reference industry")
    if ("macro" in text or "宏观" in text) and "macro_cycle_score" not in code_lower:
        issues.append("semantic mismatch: macro-timed formula must reference macro_cycle_score")
    return issues


def _today_yyyymmdd() -> str:
    return datetime.now(tz=SHANGHAI_TZ).strftime("%Y%m%d")


def _date_span_days(start: str, end: str) -> int:
    return abs((_parse_factor_date(end) - _parse_factor_date(start)).days)


def _parse_factor_date(value: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)


def _render_factor_test_code(name: str) -> str:
    return f'''"""Tests for candidate factor: {name}."""

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("factor.py")
SPEC = importlib.util.spec_from_file_location("candidate_factor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compute = MODULE.compute


def test_compute_returns_series():
    bars = pd.DataFrame({{
        "symbol": ["A", "A", "A", "B", "B", "B"],
        "trade_date": pd.date_range("2024-01-01", periods=6),
        "close": [10, 11, 12, 20, 21, 22],
    }})
    result = compute(bars)
    assert isinstance(result, pd.Series)
    assert len(result) == len(bars)
    assert result.isna().sum() > 0  # lookback creates NaN


def test_empty_input():
    result = compute(pd.DataFrame(columns=["symbol", "trade_date", "close"]))
    assert len(result) == 0
'''
