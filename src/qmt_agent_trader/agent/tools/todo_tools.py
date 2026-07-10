"""Todo-list tools for session-scoped agent progress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.todos import TodoListStore
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    ExecutionStatus,
    RecommendationStatus,
)
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.persistence.paths import PersistencePaths


def build_todo_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    paths = PersistencePaths.from_settings(deps.settings)
    store = TodoListStore(
        paths.data_root / "todos",
        locks_root=paths.locks_root,
        quarantine_root=paths.quarantine_root / "todos",
    )

    def _set_list(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        items = input_data.get("items", [])
        if not isinstance(items, list):
            return {"status": "INVALID_REQUEST", "message": "items must be a list"}
        record = store.replace_items(
            _session_id(context),
            [item if isinstance(item, dict) else {"title": str(item)} for item in items],
            goal=input_data.get("goal"),
            expected_revision=input_data.get("expected_revision"),
        )
        return _result(record)

    def _add_item(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        record = store.add_item(
            _session_id(context),
            title=str(input_data.get("title", "")),
            notes=str(input_data.get("notes", "")),
            expected_revision=input_data.get("expected_revision"),
        )
        return _result(record)

    def _update_item(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        record = store.update_item(
            _session_id(context),
            str(input_data.get("item_id", "")),
            status=input_data.get("status"),
            title=_optional_str(input_data.get("title")),
            notes=_optional_str(input_data.get("notes")),
            expected_revision=input_data.get("expected_revision"),
        )
        return _result(
            record,
            todo_consistency=_todo_consistency(input_data),
        )

    def _get_status(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        include_completed = bool(input_data.get("include_completed", True))
        return _result(store.get(_session_id(context)), include_completed=include_completed)

    def _clear_completed(
        _input_data: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        return _result(store.clear_completed(
            _session_id(context), expected_revision=_input_data.get("expected_revision")))

    return [
        _todo_tool(
            "todo_set_list",
            "创建或替换当前会话的 todo-list，用于多步骤研究任务进度跟踪。",
            {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "notes": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "PENDING",
                                        "IN_PROGRESS",
                                        "COMPLETED",
                                        "BLOCKED",
                                    ],
                                },
                            },
                            "required": ["title"],
                        },
                    },
                },
                "required": ["items"],
            },
            _set_list,
        ),
        _todo_tool(
            "todo_add_item",
            "向当前会话 todo-list 追加一个任务。",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "notes": {"type": "string"},
                    "expected_revision": {"type": "integer", "minimum": 0},
                },
                "required": ["title"],
            },
            _add_item,
        ),
        _todo_tool(
            "todo_update_item",
            "更新当前会话 todo-list 中的任务标题、备注或状态。",
            {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED"],
                    },
                    "title": {"type": "string"},
                    "notes": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "expected_revision": {"type": "integer", "minimum": 0},
                },
                "required": ["item_id"],
            },
            _update_item,
        ),
        _todo_tool(
            "todo_get_status",
            "读取当前会话 todo-list 状态。",
            {
                "type": "object",
                "properties": {
                    "include_completed": {"type": "boolean"},
                },
            },
            _get_status,
        ),
        _todo_tool(
            "todo_clear_completed",
            "清理当前会话 todo-list 中已完成的任务。",
            {
                "type": "object",
                "properties": {
                    "expected_revision": {"type": "integer", "minimum": 0},
                },
            },
            _clear_completed,
        ),
    ]


def _todo_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
) -> AgentTool:
    return tool(
        ToolSpec(
            name=name,
            description=description,
            permission=PermissionLevel.RESEARCH_WRITE,
            side_effect_level="write_generated",
            input_schema=input_schema,
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "todo_state": {"type": "object"},
                },
            },
            deterministic=False,
            timeout_seconds=10,
        ),
        fn=fn,
    )


def _session_id(context: ToolContext) -> str:
    return context.session_id or context.run_id


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _result(
    record: Any,
    *,
    include_completed: bool = True,
    todo_consistency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings = list((todo_consistency or {}).get("warnings", []))
    payload = {
        "status": "ok",
        "execution_status": ExecutionStatus.OK.value,
        "domain_status": DomainStatus.WARN.value if warnings else DomainStatus.UNKNOWN.value,
        "evidence_status": EvidenceStatus.WEAK.value if warnings else EvidenceStatus.UNKNOWN.value,
        "recommendation_status": RecommendationStatus.UNKNOWN.value,
        "warnings": warnings,
        "todo_consistency_status": (todo_consistency or {}).get(
            "todo_consistency_status",
            "UNKNOWN",
        ),
        "todo_state": record.to_payload(include_completed=include_completed),
    }
    return payload


def _todo_consistency(input_data: dict[str, Any]) -> dict[str, Any]:
    status = str(input_data.get("status") or "")
    if status != "COMPLETED":
        return {"todo_consistency_status": "UNKNOWN", "warnings": []}
    notes = str(input_data.get("notes") or "")
    evidence_refs = input_data.get("evidence_refs")
    warnings: list[str] = []
    consistency = "OK"
    failure_markers = (
        "FAIL",
        "FAILED",
        "BLOCKED",
        "NO_DATA",
        "PARTIAL_COVERAGE",
        "INVALID",
        "缺失",
        "失败",
        "阻塞",
    )
    if any(marker in notes for marker in failure_markers):
        consistency = "CONFLICT"
        warnings.append("TODO_COMPLETED_WITH_FAILED_EVIDENCE")
    elif not isinstance(evidence_refs, list) or not evidence_refs:
        consistency = "UNVERIFIED"
        warnings.append("TODO_COMPLETION_UNVERIFIED")
    return {"todo_consistency_status": consistency, "warnings": warnings}
