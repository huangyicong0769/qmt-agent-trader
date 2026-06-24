"""Experiment store API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from qmt_agent_trader.agent.errors import ExperimentNotFoundError
from qmt_agent_trader.agent.schemas import ExperimentRecord
from qmt_agent_trader.web.routes.workflows import get_experiment_store
from qmt_agent_trader.web.schemas import ExperimentDetail, ExperimentSummary

router = APIRouter()


@router.get("/", response_model=list[ExperimentSummary])
async def list_experiments(
    query: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> list[ExperimentSummary]:
    tags = [tag] if tag else None
    records = get_experiment_store().search_experiments(query=query, tags=tags, limit=limit)
    return [_summary(record) for record in records]


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: str) -> ExperimentDetail:
    try:
        record = get_experiment_store().get_experiment(experiment_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _detail(record)


def _summary(record: ExperimentRecord) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=record.experiment_id,
        kind=record.kind,
        status=record.status.value,
        created_at=record.created_at,
        updated_at=record.updated_at,
        tags=record.tags,
        artifact_count=len(record.artifacts),
    )


def _detail(record: ExperimentRecord) -> ExperimentDetail:
    return ExperimentDetail(
        **_summary(record).model_dump(),
        hypothesis=record.hypothesis,
        artifacts=record.artifacts,
        metrics=record.metrics,
        lessons=record.lessons,
    )
