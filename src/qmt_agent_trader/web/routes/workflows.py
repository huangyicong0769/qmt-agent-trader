"""Agent workflow API routes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.schemas import ExperimentRecord
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.workflows.factor_discovery import FactorDiscoveryWorkflow
from qmt_agent_trader.agent.workflows.self_bootstrap import SelfBootstrapWorkflow
from qmt_agent_trader.agent.workflows.strategy_engineering import StrategyEngineeringWorkflow
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, event_bus
from qmt_agent_trader.web.runtime import get_agent_runtime
from qmt_agent_trader.web.schemas import WorkflowRunRequest, WorkflowRunResponse

router = APIRouter()

_workflow_runs: dict[str, WorkflowRunResponse] = {}


@lru_cache
def get_experiment_store() -> ExperimentStore:
    settings = get_settings()
    return ExperimentStore(settings.resolved_data_dir / "experiments")


@router.post("/factor-discovery", response_model=WorkflowRunResponse)
async def start_factor_discovery(request: WorkflowRunRequest) -> WorkflowRunResponse:
    if not request.theme:
        raise HTTPException(status_code=422, detail="theme is required")
    run_id = new_id("run")
    return await _run_workflow(
        run_id=run_id,
        workflow_type="factor_discovery",
        runner=lambda registry, store: FactorDiscoveryWorkflow(registry, store).run(
            request.theme or "",
            request.universe,
            request.start_date,
            request.end_date,
        ),
    )


@router.post("/strategy-engineering", response_model=WorkflowRunResponse)
async def start_strategy_engineering(request: WorkflowRunRequest) -> WorkflowRunResponse:
    if not request.strategy_idea:
        raise HTTPException(status_code=422, detail="strategy_idea is required")
    run_id = new_id("run")
    return await _run_workflow(
        run_id=run_id,
        workflow_type="strategy_engineering",
        runner=lambda registry, store: StrategyEngineeringWorkflow(registry, store).run(
            request.strategy_idea or "",
            request.selected_factors,
            request.universe,
            request.start_date,
            request.end_date,
        ),
    )


@router.post("/self-bootstrap", response_model=WorkflowRunResponse)
async def start_self_bootstrap(request: WorkflowRunRequest) -> WorkflowRunResponse:
    run_id = new_id("run")
    return await _run_workflow(
        run_id=run_id,
        workflow_type="self_bootstrap",
        runner=lambda registry, store: SelfBootstrapWorkflow(registry, store).run(
            request.recent_experiment_ids
        ),
    )


@router.get("/runs/{run_id}", response_model=WorkflowRunResponse)
async def get_run_status(run_id: str) -> WorkflowRunResponse:
    run = _workflow_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="workflow run not found")
    return run


async def _run_workflow(
    *,
    run_id: str,
    workflow_type: str,
    runner: WorkflowRunner,
) -> WorkflowRunResponse:
    started = WorkflowRunResponse(
        run_id=run_id,
        workflow_type=workflow_type,
        status="RUNNING",
        message="Workflow started.",
    )
    _workflow_runs[run_id] = started
    await event_bus.publish(
        AgentEvent(
            run_id=run_id,
            event_type=AgentEventType.RUN_STARTED,
            title=f"Workflow started: {workflow_type}",
            message="Workflow started.",
        )
    )
    try:
        record = await asyncio.to_thread(
            runner,
            get_agent_runtime().agent_registry(),
            get_experiment_store(),
        )
    except Exception as exc:
        failed = WorkflowRunResponse(
            run_id=run_id,
            workflow_type=workflow_type,
            status="FAILED",
            message=str(exc),
        )
        _workflow_runs[run_id] = failed
        await event_bus.publish(
            AgentEvent(
                run_id=run_id,
                event_type=AgentEventType.RUN_FAILED,
                title=f"Workflow failed: {workflow_type}",
                message=str(exc),
            )
        )
        return failed

    response = WorkflowRunResponse(
        run_id=run_id,
        workflow_type=workflow_type,
        status=record.status.value,
        experiment_id=record.experiment_id,
        message="Workflow completed.",
        result=record.model_dump(mode="json"),
    )
    _workflow_runs[run_id] = response
    await event_bus.publish(
        AgentEvent(
            run_id=run_id,
            experiment_id=record.experiment_id,
            event_type=AgentEventType.RUN_COMPLETED,
            title=f"Workflow completed: {workflow_type}",
            message=record.status.value,
            payload={"experiment_id": record.experiment_id},
        )
    )
    return response


WorkflowRunner = Callable[[AgentToolRegistry, ExperimentStore], ExperimentRecord]
