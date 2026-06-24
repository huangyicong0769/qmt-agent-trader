"""Experiment tools: log_experiment_event, search_experiments."""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.agent.errors import ExperimentNotFoundError
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool

_store: ExperimentStore | None = None


def set_experiment_store(store: ExperimentStore) -> None:
    global _store
    _store = store


def _get_store() -> ExperimentStore:
    if _store is None:
        raise RuntimeError("experiment store not wired")
    return _store


# ── log_experiment_event ─────────────────────────────────────────────────────


def _log_experiment_event(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    store = _get_store()
    exp_id: str = str(input_data.get("experiment_id", ""))
    event_type = input_data.get("event_type", "observation")
    message = input_data.get("message", "")
    input_data.get("metadata", {})

    if not exp_id:
        return {"status": "error", "message": "experiment_id is required"}

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
