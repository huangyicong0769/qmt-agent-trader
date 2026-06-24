"""Tool discovery and execution API routes."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from qmt_agent_trader.agent.errors import ToolExecutionError, ToolNotFoundError
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.errors import PermissionDeniedError
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, event_bus
from qmt_agent_trader.web.schemas import ToolRunRequest, ToolRunResponse

router = APIRouter()

WEB_BLOCKED_PERMISSIONS = {
    PermissionLevel.APPROVAL_REQUIRED,
    PermissionLevel.FORBIDDEN_TO_LLM,
}


@lru_cache
def get_registry() -> AgentToolRegistry:
    settings = get_settings()
    data_lake = DataLake(
        root=settings.resolved_data_dir / "lake",
        duckdb_path=settings.resolved_data_dir / "qmt_agent_trader.duckdb",
    )
    return build_agent_registry(
        data_lake=data_lake,
        audit_path=settings.resolved_log_dir / "audit" / "agent_tool_calls.jsonl",
        experiment_root=settings.resolved_data_dir / "experiments",
        sandbox=CodeSandbox(),
    )


@router.get("/")
async def list_tools(permission: str | None = None) -> list[dict[str, object]]:
    return get_registry().list_tools(permission=permission)


@router.get("/{tool_name}", response_model=ToolSpec)
async def get_tool(tool_name: str) -> ToolSpec:
    try:
        return get_registry().describe_tool(tool_name)
    except ToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{tool_name}/run", response_model=ToolRunResponse)
async def run_tool(tool_name: str, request: ToolRunRequest) -> ToolRunResponse:
    registry = get_registry()
    run_id = new_id("run")
    try:
        spec = registry.describe_tool(tool_name)
    except ToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    await event_bus.publish(
        AgentEvent(
            run_id=run_id,
            experiment_id=request.experiment_id,
            event_type=AgentEventType.TOOL_CALL_STARTED,
            title=f"Tool started: {tool_name}",
            message=spec.description,
            payload={"tool_name": tool_name, "permission": spec.permission.value},
        )
    )

    if spec.permission in WEB_BLOCKED_PERMISSIONS:
        message = f"tool '{tool_name}' is display-only in the web UI"
        _audit_denial(registry, spec, run_id, request, message)
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                experiment_id=request.experiment_id,
                event_type=AgentEventType.TOOL_PERMISSION_DENIED,
                title=f"Tool denied: {tool_name}",
                message=message,
                payload={"tool_name": tool_name, "permission": spec.permission.value},
            )
        )
        return ToolRunResponse(
            run_id=run_id,
            tool_name=tool_name,
            status="permission_denied",
            error_message=message,
        )

    context = ToolContext(
        run_id=run_id,
        experiment_id=request.experiment_id,
        requested_by_llm=False,
        dry_run=request.dry_run,
        user_id=request.user_id,
    )
    try:
        result = registry.run_tool(tool_name, request.input_data, context)
    except PermissionDeniedError as exc:
        message = str(exc)
        _audit_denial(registry, spec, run_id, request, message)
        await _publish_tool_failure(run_id, request.experiment_id, tool_name, message, True)
        return ToolRunResponse(
            run_id=run_id,
            tool_name=tool_name,
            status="permission_denied",
            error_message=message,
        )
    except ToolExecutionError as exc:
        message = str(exc)
        await _publish_tool_failure(run_id, request.experiment_id, tool_name, message, False)
        return ToolRunResponse(
            run_id=run_id,
            tool_name=tool_name,
            status="error",
            result={"error": True},
            error_message=message,
        )

    await event_bus.publish(
        AgentEvent(
            run_id=run_id,
            experiment_id=request.experiment_id,
            event_type=AgentEventType.TOOL_CALL_COMPLETED,
            title=f"Tool completed: {tool_name}",
            message="Tool completed successfully.",
            payload={"tool_name": tool_name},
        )
    )
    return ToolRunResponse(run_id=run_id, tool_name=tool_name, status="ok", result=result)


def _audit_denial(
    registry: AgentToolRegistry,
    spec: ToolSpec,
    run_id: str,
    request: ToolRunRequest,
    error_message: str,
) -> None:
    if registry.audit_logger is None:
        return
    registry.audit_logger.append(
        tool_name=spec.name,
        run_id=run_id,
        experiment_id=request.experiment_id,
        permission=spec.permission.value,
        requested_by_llm=False,
        input_data=request.input_data,
        output_data={"error": True, "message": error_message},
        status="permission_denied",
        error_message=error_message,
        duration_ms=0,
    )


async def _publish_tool_failure(
    run_id: str,
    experiment_id: str | None,
    tool_name: str,
    message: str,
    permission_denied: bool,
) -> None:
    await event_bus.publish(
        AgentEvent(
            run_id=run_id,
            experiment_id=experiment_id,
            event_type=(
                AgentEventType.TOOL_PERMISSION_DENIED
                if permission_denied
                else AgentEventType.TOOL_CALL_FAILED
            ),
            title=f"Tool failed: {tool_name}",
            message=message,
            payload={"tool_name": tool_name},
        )
    )
