"""Experiment tools: log_experiment_event, search_experiments."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.errors import ExperimentNotFoundError
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import AgentTool, tool

_store: ExperimentStore | None = None
_store_var: ContextVar[ExperimentStore | None] = ContextVar(
    "experiment_tool_store",
    default=None,
)
_audit_logger: AuditLogger | None = None
_audit_logger_var: ContextVar[AuditLogger | None] = ContextVar(
    "experiment_tool_audit_logger",
    default=None,
)


def set_experiment_store(store: ExperimentStore) -> None:
    global _store
    _store = store


def _get_store() -> ExperimentStore:
    store = _store_var.get() or _store
    if store is None:
        raise RuntimeError("experiment store not wired")
    return store


def _get_audit_log_path() -> Path | None:
    logger = _audit_logger_var.get() or _audit_logger
    return logger.log_path if logger is not None else None


def _with_deps(
    deps: AgentToolDependencies,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    token = _store_var.set(deps.experiment_store)
    audit_token = _audit_logger_var.set(deps.audit_logger)
    try:
        return fn(input_data, context)
    finally:
        _audit_logger_var.reset(audit_token)
        _store_var.reset(token)


# ── log_experiment_event ─────────────────────────────────────────────────────


def _log_experiment_event(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    store = _get_store()
    exp_id: str = str(input_data.get("experiment_id") or context.experiment_id or "")
    event_type = input_data.get("event_type", "observation")
    message = input_data.get("message", "")
    input_data.get("metadata", {})

    if not exp_id:
        return {"status": "error", "message": "experiment_id or context experiment_id is required"}

    try:
        store.get_experiment(exp_id)
    except ExperimentNotFoundError:
        return {"status": "error", "message": f"experiment '{exp_id}' not found"}

    store.add_lesson(exp_id, f"[{event_type}] {message}")
    return {"status": "logged", "experiment_id": exp_id}


log_experiment_event_tool: AgentTool = tool(
    ToolSpec(
        name="log_experiment_event",
        description="记录一次实验事件或观察。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_log_experiment_event,
)

# ── search_experiments ───────────────────────────────────────────────────────


def _search_experiments(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    store = _get_store()
    query = input_data.get("query")
    tags = input_data.get("tags")
    if query is None and tags is None and context.session_id:
        tags = [f"session:{context.session_id}"]
    limit = input_data.get("limit", 20)
    results = store.search_experiments(query=query, tags=tags, limit=limit)
    return {
        "experiments": [
            {
                "experiment_id": r.experiment_id,
                "kind": r.kind,
                "status": r.status.value,
                "tags": r.tags,
                "lessons": r.lessons[-5:] if r.lessons else [],
                "created_at": r.created_at.isoformat(),
            }
            for r in results
        ]
    }


search_experiments_tool: AgentTool = tool(
    ToolSpec(
        name="search_experiments",
        description="搜索过去实验记录，避免重复试错。",
        permission=PermissionLevel.READ_ONLY,
        deterministic=True,
    ),
    fn=_search_experiments,
)

# ── get_experiment_tool_calls ────────────────────────────────────────────────


def _get_experiment_tool_calls(
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    audit_path = _get_audit_log_path()
    if audit_path is None or not audit_path.exists():
        return {"status": "NOT_AVAILABLE", "message": "audit log is not available"}
    explicit_experiment_id = input_data.get("experiment_id")
    experiment_id = str(explicit_experiment_id or "")
    run_id = str(input_data.get("run_id") or "")
    session_id = _normalize_session_id(input_data.get("session_id"), context)
    if not experiment_id and not run_id:
        session_id = session_id or str(context.session_id or "")
    if not experiment_id and not run_id and not session_id and context.experiment_id:
        experiment_id = context.experiment_id
    limit = int(input_data.get("limit") or 50)
    if not experiment_id and not run_id and not session_id:
        return {"status": "error", "message": "experiment_id, run_id, or session_id is required"}

    calls: list[dict[str, Any]] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if experiment_id and entry.get("experiment_id") != experiment_id:
            continue
        if run_id and entry.get("run_id") != run_id:
            continue
        if session_id and entry.get("session_id") != session_id:
            continue
        if entry.get("status") == "started":
            continue
        calls.append(_summarize_audit_entry(entry))
    return {
        "status": "ok",
        "experiment_id": experiment_id or None,
        "run_id": run_id or None,
        "session_id": session_id or None,
        "count": len(calls),
        "tool_calls": calls[-limit:],
    }


get_experiment_tool_calls_tool: AgentTool = tool(
    ToolSpec(
        name="get_experiment_tool_calls",
        description=(
            "按 experiment_id 或 run_id 查询真实工具调用审计记录，"
            "用于复盘刚才实际调用了哪些工具。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
                "run_id": {"type": "string"},
                "session_id": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=True,
    ),
    fn=_get_experiment_tool_calls,
)


def _summarize_audit_entry(entry: dict[str, Any]) -> dict[str, Any]:
    output = entry.get("output_data")
    return {
        "timestamp": entry.get("timestamp"),
        "run_id": entry.get("run_id"),
        "session_id": entry.get("session_id"),
        "experiment_id": entry.get("experiment_id"),
        "tool_name": entry.get("tool_name"),
        "status": entry.get("status"),
        "error_message": entry.get("error_message"),
        "duration_ms": entry.get("duration_ms"),
        "output": _compact_output(output if isinstance(output, dict) else {}),
    }


def _compact_output(output: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "message",
        "run_id",
        "strategy_id",
        "factor_name",
        "factor_ids",
        "execution_backend",
        "research_only",
        "live_trading_allowed",
        "code_path",
        "tests_path",
        "report_path",
        "metrics",
        "diagnostics",
        "data_window",
    ]
    compact = {key: output[key] for key in keys if key in output}
    if "tools" in output:
        compact["tool_count"] = len(output["tools"]) if isinstance(output["tools"], list) else None
    if "strategies" in output:
        compact["strategy_count"] = (
            len(output["strategies"]) if isinstance(output["strategies"], list) else None
        )
    return compact


def _normalize_session_id(value: Any, context: ToolContext) -> str:
    text = str(value or "").strip()
    if text.lower() in {"current", "current_session", "this_session", "当前", "当前会话"}:
        return str(context.session_id or "")
    return text


def build_experiment_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    return [
        tool(
            log_experiment_event_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _log_experiment_event, input_data, context
            ),
        ),
        tool(
            search_experiments_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _search_experiments, input_data, context
            ),
        ),
        tool(
            get_experiment_tool_calls_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _get_experiment_tool_calls, input_data, context
            ),
        ),
    ]
