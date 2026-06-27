"""Tool discovery and execution API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from qmt_agent_trader.agent.errors import ToolExecutionError, ToolNotFoundError
from qmt_agent_trader.agent.permissions import ToolCallMode
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.core.errors import PermissionDeniedError
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, event_bus
from qmt_agent_trader.web.runtime import get_agent_runtime
from qmt_agent_trader.web.schemas import ToolRunRequest, ToolRunResponse

router = APIRouter()


@router.get("/")
async def list_tools(permission: str | None = None) -> list[dict[str, object]]:
    return get_agent_runtime().list_tools(
        permission=permission,
        agent_callable_only=True,
        call_mode=ToolCallMode.AUTONOMOUS_AGENT,
    )


@router.get("/{tool_name}", response_model=ToolSpec)
async def get_tool(tool_name: str) -> ToolSpec:
    try:
        return get_agent_runtime().describe_tool(tool_name)
    except ToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{tool_name}/run", response_model=ToolRunResponse)
async def run_tool(tool_name: str, request: ToolRunRequest) -> ToolRunResponse:
    runtime = get_agent_runtime()
    run_id = new_id("run")
    try:
        spec = runtime.describe_tool(tool_name)
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

    context = ToolContext(
        run_id=run_id,
        experiment_id=request.experiment_id,
        requested_by_llm=True,
        call_mode=ToolCallMode.AUTONOMOUS_AGENT,
        dry_run=request.dry_run,
        user_id=request.user_id,
    )
    input_data = dict(request.input_data)
    if tool_name == "run_remote_data_update" and request.dry_run:
        input_data.setdefault("dry_run", True)
    try:
        result = runtime.run_tool(tool_name, input_data, context)
    except PermissionDeniedError as exc:
        message = str(exc)
        _audit_denial(runtime, spec, run_id, request, message)
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
    runtime: object,
    spec: ToolSpec,
    run_id: str,
    request: ToolRunRequest,
    error_message: str,
) -> None:
    registry = getattr(runtime, "agent_registry", lambda: None)()
    audit_logger = getattr(registry, "audit_logger", None)
    if audit_logger is None:
        return
    audit_logger.append(
        tool_name=spec.name,
        run_id=run_id,
        experiment_id=request.experiment_id,
        permission=spec.permission.value,
        requested_by_llm=True,
        call_mode=ToolCallMode.AUTONOMOUS_AGENT.value,
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
