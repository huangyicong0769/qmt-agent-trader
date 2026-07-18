"""In-process event bus for Agent Studio progress streams."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from qmt_agent_trader.core.ids import new_id


class AgentEventType(StrEnum):
    RUN_STARTED = "RUN_STARTED"
    RUN_CANCELLING = "RUN_CANCELLING"
    RUN_CANCELLED = "RUN_CANCELLED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    RUN_DIAGNOSTIC = "RUN_DIAGNOSTIC"
    LLM_MESSAGE = "LLM_MESSAGE"
    LLM_TOKEN_DELTA = "LLM_TOKEN_DELTA"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_COMPLETED = "TOOL_CALL_COMPLETED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    TOOL_PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"
    TODO_STATUS_UPDATED = "TODO_STATUS_UPDATED"
    ARTIFACT_CREATED = "ARTIFACT_CREATED"
    EXPERIMENT_UPDATED = "EXPERIMENT_UPDATED"
    SANDBOX_CHECK_STARTED = "SANDBOX_CHECK_STARTED"
    SANDBOX_CHECK_COMPLETED = "SANDBOX_CHECK_COMPLETED"
    SANDBOX_CHECK_FAILED = "SANDBOX_CHECK_FAILED"
    PROGRESS = "PROGRESS"


class AgentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    experiment_id: str | None = None
    event_type: AgentEventType
    title: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Publish/subscribe event bus for real-time agent progress."""

    def __init__(self, *, history_limit: int = 512) -> None:
        if history_limit < 8:
            raise ValueError("history_limit must be at least 8")
        self._queues: dict[str, list[asyncio.Queue[AgentEvent]]] = defaultdict(list)
        self._history: dict[str, deque[AgentEvent]] = defaultdict(
            lambda: deque(maxlen=history_limit)
        )

    async def publish(self, event: AgentEvent) -> None:
        self._history[event.run_id].append(event)
        for queue in tuple(self._queues.get(event.run_id, [])):
            await queue.put(event)
        for queue in tuple(self._queues.get("*", [])):
            await queue.put(event)

    async def subscribe(self, run_id: str) -> asyncio.Queue[AgentEvent]:
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._queues[run_id].append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[AgentEvent]) -> None:
        queues = self._queues.get(run_id)
        if not queues:
            return
        self._queues[run_id] = [item for item in queues if item is not queue]
        if not self._queues[run_id]:
            self._queues.pop(run_id, None)

    def clear_history(self, run_id: str) -> None:
        self._history.pop(run_id, None)

    def get_history(self, run_id: str) -> list[AgentEvent]:
        return list(self._history.get(run_id, []))


event_bus = EventBus()
