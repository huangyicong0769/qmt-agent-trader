"""Application-scoped lifecycle management for chat agent runs.

This module deliberately has no NiceGUI dependency.  A run is owned by the
application process, while pages and SSE clients are disposable subscribers.
Run task recovery is intentionally limited to the same service process; a
process restart does not reconstruct an in-flight worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from qmt_agent_trader.agent.cancellation import CancellationToken
from qmt_agent_trader.agent.orchestrator import AgentOrchestrator, OrchestratorEvent
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.web.chat_repository import (
    ChatSessionRepository,
    build_chat_session_repository,
)
from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, EventBus, event_bus
from qmt_agent_trader.web.schemas import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

DEFAULT_EVENT_HISTORY_LIMIT = 256
DEFAULT_TERMINAL_RUN_TTL_SECONDS = 300.0

PERSISTED_EVENT_TYPES = frozenset(
    {
        "user_message",
        "run_started",
        "cancelling",
        "cancelled",
        "tool_start",
        "tool_args",
        "tool_done",
        "final_message",
        "done",
        "error",
    }
)

# Lock order is manager._lock -> run.event_lock when both are needed.  The
# per-Run lock serializes every state transition, sequence allocation,
# persistence, history append, and subscriber broadcast.  Once a terminal
# event is appended, its snapshot state is already terminal and no later event
# may be appended.  Emit operations acquire only run.event_lock, so slow
# persistence for one Run never serializes unrelated Sessions.  No code may
# acquire manager._lock while holding a run.event_lock.


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


TERMINAL_STATUSES = frozenset(
    {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}
)


class ChatRunError(RuntimeError):
    """Base class for run lifecycle errors."""


class RunAlreadyActiveError(ChatRunError):
    """Raised when a session already has a non-terminal run."""


class SuccessorAlreadyPendingError(ChatRunError):
    """Raised when an interrupt already has a successor request."""


class SessionDeletionBlockedError(ChatRunError):
    """Raised when a session still has owned work or a successor request."""


class InvalidRunTransition(ChatRunError):
    """Raised when a run state transition violates the state machine."""


@dataclass(frozen=True)
class RunEvent:
    sequence: int
    run_id: str
    session_id: str
    event_type: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=shanghai_now_iso)
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            # Keep the legacy OrchestratorEvent SSE field available while
            # exposing the ordered RunEvent name used by new clients.
            "type": self.event_type,
            "message": self.message,
            "data": self.data,
            "created_at": self.created_at,
            "terminal": self.terminal,
        }

    def to_sse(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    session_id: str
    status: RunStatus
    message: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    last_event_sequence: int
    cancellation_requested: bool
    accumulated_draft: str
    accumulated_draft_through_sequence: int
    recent_tool: dict[str, Any] | None
    successor_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "last_event_sequence": self.last_event_sequence,
            "cancellation_requested": self.cancellation_requested,
            "accumulated_draft": self.accumulated_draft,
            "accumulated_draft_through_sequence": self.accumulated_draft_through_sequence,
            "recent_tool": self.recent_tool,
            "successor_run_id": self.successor_run_id,
        }


@dataclass
class _SuccessorRequest:
    request_id: str
    message: str
    persist_user_message: bool = True


@dataclass
class _ChatRun:
    run_id: str
    session_id: str
    message: str
    history: list[dict[str, Any]]
    token: CancellationToken = field(default_factory=CancellationToken)
    status: RunStatus = RunStatus.PENDING
    created_at: str = field(default_factory=shanghai_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    last_event_sequence: int = 0
    accumulated_draft: str = ""
    accumulated_draft_through_sequence: int = 0
    recent_tool: dict[str, Any] | None = None
    task: asyncio.Task[None] | None = None
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    history_events: deque[RunEvent] = field(default_factory=deque)
    subscribers: set[asyncio.Queue[RunEvent]] = field(default_factory=set)
    event_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    persistence_failure_reported: bool = False
    terminal_event_sequence: int | None = None


class ChatRunManager:
    """Own background agent tasks and expose ordered, replayable run events."""

    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator | Any | None = None,
        repository: ChatSessionRepository | None = None,
        bus: EventBus | None = None,
        history_limit: int = DEFAULT_EVENT_HISTORY_LIMIT,
        terminal_ttl_seconds: float = DEFAULT_TERMINAL_RUN_TTL_SECONDS,
    ) -> None:
        if history_limit < 8:
            raise ValueError("history_limit must be at least 8")
        if terminal_ttl_seconds <= 0:
            raise ValueError("terminal_ttl_seconds must be positive")
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.repository = repository or build_chat_session_repository()
        self.event_bus = bus or event_bus
        self.history_limit = history_limit
        self.terminal_ttl_seconds = terminal_ttl_seconds
        self._runs: dict[str, _ChatRun] = {}
        self._active_by_session: dict[str, str] = {}
        self._successors: dict[str, _SuccessorRequest] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def get_run(self, run_id: str) -> RunSnapshot | None:
        run = self._runs.get(run_id)
        return self._snapshot(run) if run is not None else None

    def get_active_run(self, session_id: str) -> RunSnapshot | None:
        run_id = self._active_by_session.get(session_id)
        run = self._runs.get(run_id) if run_id is not None else None
        if run is None or not self._is_execution_active(run):
            return None
        return self._snapshot(run)

    def has_pending_successor(self, session_id: str) -> bool:
        return session_id in self._successors

    async def delete_session(self, session_id: str) -> bool:
        """Delete only after the manager serializes the lifecycle check."""
        async with self._session_lock(session_id):
            if self._active_run(session_id) is not None or self.has_pending_successor(
                session_id
            ):
                raise SessionDeletionBlockedError(
                    "当前会话仍在运行或停止中，请等待任务结束后再删除。"
                )
            return self.repository.delete(session_id)

    def subscriber_count(self, run_id: str) -> int:
        run = self._runs.get(run_id)
        return len(run.subscribers) if run is not None else 0

    async def start_run(
        self,
        session_id: str,
        message: str,
        *,
        persist_user_message: bool = True,
    ) -> RunSnapshot:
        async with self._session_lock(session_id):
            if session_id in self._successors:
                raise SuccessorAlreadyPendingError(
                    f"session {session_id} already has a pending successor"
                )
            active = self._active_run(session_id)
            if active is not None:
                raise RunAlreadyActiveError(
                    f"session {session_id} already has active run {active.run_id}"
                )
            return await self._start_run_locked(
                session_id,
                message,
                persist_user_message=persist_user_message,
            )

    async def interrupt_and_start(
        self,
        session_id: str,
        message: str,
        *,
        persist_user_message: bool = True,
    ) -> RunSnapshot:
        async with self._session_lock(session_id):
            if session_id in self._successors:
                raise SuccessorAlreadyPendingError(
                    f"session {session_id} already has a pending successor"
                )
            active = self._active_run(session_id)
            if active is None:
                return await self._start_run_locked(
                    session_id,
                    message,
                    persist_user_message=persist_user_message,
                )
            await self._request_cancel_serialized(active)
            request = _SuccessorRequest(
                request_id=new_id("successor"),
                message=message,
                persist_user_message=persist_user_message,
            )
            if self._is_execution_active(active):
                # The old worker still owns this Session until teardown.  Keep
                # the marker only for that handoff window, after cancellation
                # has conclusively been serialized with its event stream.
                self._successors[session_id] = request
                return self._snapshot(active)
            return await self._start_run_locked(
                session_id,
                message,
                persist_user_message=persist_user_message,
                run_id=request.request_id,
            )

    async def request_cancel(self, run_id: str) -> RunSnapshot | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        async with self._session_lock(run.session_id):
            current = self._runs.get(run_id)
            if current is None:
                return None
            return await self._request_cancel_serialized(current)

    async def wait_for_run(self, run_id: str) -> RunSnapshot:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        await run.completion_event.wait()
        if run.task is not None and run.task is not asyncio.current_task():
            await run.task
        return self._snapshot(run)

    async def subscribe(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncGenerator[RunEvent, None]:
        run = self._runs.get(run_id)
        if run is None:
            return
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        async with self._lock:
            async with run.event_lock:
                current = self._runs.get(run_id)
                if current is None or current is not run:
                    return
                snapshot_data = self._snapshot(run).to_dict()
                snapshot_is_terminal = run.status in TERMINAL_STATUSES
                snapshot = RunEvent(
                    sequence=0,
                    run_id=run.run_id,
                    session_id=run.session_id,
                    event_type="snapshot",
                    data={"snapshot": snapshot_data},
                )
                draft_through = int(
                    snapshot_data.get("accumulated_draft_through_sequence", 0)
                )
                replay = [
                    event
                    for event in run.history_events
                    if event.sequence > after_sequence
                    and not (
                        event.event_type == "token"
                        and event.sequence <= draft_through
                    )
                ]
                capture_sequence = run.last_event_sequence
                run.subscribers.add(queue)

        cursor = after_sequence
        try:
            yield snapshot
            for event in replay:
                if event.sequence <= cursor:
                    continue
                cursor = event.sequence
                yield event
                if is_terminal_run_event(event):
                    return
            if snapshot_is_terminal:
                # The snapshot and replay were captured while event_lock was
                # held.  A terminal snapshot therefore already includes the
                # terminal event in replay (unless the caller's cursor covers
                # it), and cannot wait for future producer activity.  Drain
                # only items from that capture boundary before ending.
                while True:
                    try:
                        event = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if event.sequence <= cursor or event.sequence > capture_sequence:
                        continue
                    cursor = event.sequence
                    yield event
                    if is_terminal_run_event(event):
                        return
                return
            if run.completion_event.is_set():
                # Completion can race with replay.  Drain events already
                # enqueued for this subscriber before returning so a terminal
                # event cannot disappear between replay and teardown.
                while True:
                    try:
                        event = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if event.sequence <= cursor:
                        continue
                    cursor = event.sequence
                    yield event
                    if is_terminal_run_event(event):
                        return
                return
            while True:
                event = await queue.get()
                if event.sequence <= cursor:
                    continue
                cursor = event.sequence
                yield event
                if is_terminal_run_event(event):
                    return
        finally:
            async with self._lock:
                current = self._runs.get(run_id)
                if current is run:
                    async with run.event_lock:
                        run.subscribers.discard(queue)

    async def _request_cancel_serialized(self, run: _ChatRun) -> RunSnapshot:
        """Commit cancellation intent and its event in one Run transaction."""
        async with run.event_lock:
            if run.status in TERMINAL_STATUSES:
                return self._snapshot(run)
            if run.status is RunStatus.CANCELLING:
                run.token.request_cancel()
                return self._snapshot(run)
            self._transition(run, RunStatus.CANCELLING)
            run.token.request_cancel()
            event = self._new_event(
                run,
                "cancelling",
                "正在停止，当前工具调用结束后生效。",
                {"reason": "user_request"},
            )
            if not await self._persist_or_fail(run, event):
                return self._snapshot(run)
            await self._append_and_broadcast(run, event)
            return self._snapshot(run)

    async def _start_run_locked(
        self,
        session_id: str,
        message: str,
        *,
        persist_user_message: bool = True,
        run_id: str | None = None,
    ) -> RunSnapshot:
        history = self._load_history(session_id)
        run = _ChatRun(
            run_id=run_id or new_id("run"),
            session_id=session_id,
            message=message,
            history=history,
        )
        run.history_events = deque(maxlen=self.history_limit)
        async with self._lock:
            existing = self._active_run(session_id)
            if existing is not None:
                raise RunAlreadyActiveError(
                    f"session {session_id} already has active run {existing.run_id}"
                )
            self._runs[run.run_id] = run
            self._active_by_session[session_id] = run.run_id
        if persist_user_message:
            await self._emit(
                run,
                "user_message",
                message,
                data={"content": message, "phase": "input"},
            )
        if run.status in TERMINAL_STATUSES:
            run.completion_event.set()
            self._schedule_cleanup(run.run_id)
            async with self._lock:
                if self._active_by_session.get(session_id) == run.run_id:
                    self._active_by_session.pop(session_id, None)
            return self._snapshot(run)
        run.task = asyncio.create_task(
            self._execute_run(run),
            name=f"chat-run-{run.run_id}",
        )
        return self._snapshot(run)

    def _load_history(self, session_id: str) -> list[dict[str, Any]]:
        session = self.repository.get(session_id)
        if session is None:
            raise KeyError(f"chat session not found: {session_id}")
        return [
            {"role": message.role, "content": message.content}
            for message in session.messages
        ]

    def _active_run(self, session_id: str) -> _ChatRun | None:
        run_id = self._active_by_session.get(session_id)
        run = self._runs.get(run_id) if run_id is not None else None
        if run is None or not self._is_execution_active(run):
            return None
        return run

    @staticmethod
    def _is_execution_active(run: _ChatRun) -> bool:
        """Keep a terminal run active until its manager task has torn down."""
        return run.status not in TERMINAL_STATUSES or (
            run.task is not None and not run.task.done()
        )

    async def _execute_run(self, run: _ChatRun) -> None:
        terminal_seen = False
        try:
            async with run.event_lock:
                if run.status is RunStatus.PENDING:
                    self._transition(run, RunStatus.RUNNING)
                elif run.status is not RunStatus.CANCELLING:
                    raise InvalidRunTransition(
                        f"run {run.run_id} cannot start from {run.status}"
                    )
                elif run.started_at is None:
                    run.started_at = shanghai_now_iso()
            if run.status in TERMINAL_STATUSES:
                return
            stream = self.orchestrator.execute_stream(
                message=run.message,
                run_id=run.run_id,
                session_id=run.session_id,
                history=run.history,
                cancel_requested=run.token,
            )
            try:
                async for event in stream:
                    terminal_seen = await self._handle_orchestrator_event(run, event)
                    if terminal_seen:
                        break
            finally:
                close_stream = getattr(stream, "aclose", None)
                if callable(close_stream):
                    await close_stream()
            if not terminal_seen and run.status not in TERMINAL_STATUSES:
                error = "agent run ended without a terminal event"
                await self._emit_terminal(
                    run,
                    "error",
                    error,
                    target_status=RunStatus.FAILED,
                    error=error,
                    data={"error": error},
                )
        except asyncio.CancelledError:
            logger.info("chat run task cancelled during application shutdown: %s", run.run_id)
            raise
        except Exception as exc:
            logger.exception("chat run failed: %s", run.run_id)
            if run.status not in TERMINAL_STATUSES:
                error = str(exc)
                await self._emit_terminal(
                    run,
                    "error",
                    error,
                    target_status=RunStatus.FAILED,
                    error=error,
                    data={"error": error},
                )
        finally:
            if run.status in TERMINAL_STATUSES:
                run.completion_event.set()
                await self._finish_and_maybe_start_successor(run)

    async def _handle_orchestrator_event(
        self,
        run: _ChatRun,
        event: OrchestratorEvent,
    ) -> bool:
        event_type = event.type
        data = dict(event.data)
        message = event.message
        if event_type == "run_started":
            if run.status in TERMINAL_STATUSES:
                return True
            await self._emit(run, "run_started", message, data=data)
            return run.status in TERMINAL_STATUSES
        if event_type == "cancelled":
            if run.status in TERMINAL_STATUSES:
                return True
            await self._emit_terminal(
                run,
                "cancelled",
                message or "Run cancelled.",
                target_status=RunStatus.CANCELLED,
                data=data,
            )
            return True
        if event_type == "error" and not data.get("fallback"):
            if run.status in TERMINAL_STATUSES:
                return True
            error = message or str(data.get("error", "agent run failed"))
            await self._emit_terminal(
                run,
                "error",
                error,
                target_status=RunStatus.FAILED,
                error=error,
                data=data,
            )
            return True
        if event_type == "done":
            if run.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
                return True
            if run.status is RunStatus.CANCELLING:
                # Cancellation won the Run's serialized event boundary.  An
                # already-queued completion must not turn a committed stop
                # request into RUN_COMPLETED after the client saw CANCELLING.
                await self._emit_terminal(
                    run,
                    "cancelled",
                    "Execution cancelled by user.",
                    target_status=RunStatus.CANCELLED,
                    data={
                        **data,
                        "reason": "user_interrupt",
                        "completion_suppressed": True,
                    },
                )
                return True
            await self._emit_terminal(
                run,
                "done",
                message,
                target_status=RunStatus.COMPLETED,
                data=data,
            )
            return True
        if run.status in TERMINAL_STATUSES:
            return True
        await self._emit(run, event_type, message, data=data, persist=True)
        return run.status in TERMINAL_STATUSES

    async def _emit(
        self,
        run: _ChatRun,
        event_type: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> RunEvent:
        event_data = dict(data or {})
        async with run.event_lock:
            self._assert_can_append(run, event_type)
            event = self._new_event(run, event_type, message, event_data)
            if persist and not await self._persist_or_fail(run, event):
                return event
            await self._append_and_broadcast(run, event)
            return event

    async def _emit_terminal(
        self,
        run: _ChatRun,
        event_type: str,
        message: str,
        *,
        target_status: RunStatus,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RunEvent | None:
        """Persist, commit, and publish a terminal event in one Run boundary."""
        event_data = dict(data or {})
        async with run.event_lock:
            if run.terminal_event_sequence is not None or run.status in TERMINAL_STATUSES:
                return None
            if target_status is RunStatus.CANCELLED and run.status is not RunStatus.CANCELLING:
                self._transition(run, RunStatus.CANCELLING)
            event = self._new_event(run, event_type, message, event_data)
            if not event.terminal:
                raise ValueError(f"{event_type} is not a terminal RunEvent")
            if not await self._persist_or_fail(run, event):
                return None
            if error is not None:
                run.error = error
            self._transition(run, target_status)
            await self._append_and_broadcast(run, event)
            return event

    @staticmethod
    def _new_event(
        run: _ChatRun,
        event_type: str,
        message: str,
        data: dict[str, Any],
    ) -> RunEvent:
        return RunEvent(
            sequence=run.last_event_sequence + 1,
            run_id=run.run_id,
            session_id=run.session_id,
            event_type=event_type,
            message=message,
            data=data,
            terminal=_terminal_for_event(event_type, data),
        )

    async def _persist_or_fail(self, run: _ChatRun, event: RunEvent) -> bool:
        if self._persistent_message(event) is None:
            return True
        try:
            await asyncio.to_thread(self._persist_event, event)
        except Exception as exc:
            logger.exception(
                "failed to persist chat run event %s/%s",
                run.run_id,
                run.last_event_sequence + 1,
            )
            await self._record_persistence_failure(run, exc)
            return False
        return True

    def _persist_event(self, event: RunEvent) -> None:
        persisted = self._persistent_message(event)
        if persisted is None:
            return
        role, content, metadata = persisted

        def operation(session: ChatSession) -> ChatSession:
            for existing in session.messages:
                if _same_event_marker(existing, event):
                    return session
            message = ChatMessage(
                session_id=session.session_id,
                role=role,
                content=content,
                metadata=metadata,
            )
            return session.model_copy(
                update={
                    "messages": [*session.messages, message],
                    "updated_at": shanghai_now_iso(),
                }
            )

        self.repository.update(event.session_id, operation)

    def _persistent_message(
        self,
        event: RunEvent,
    ) -> tuple[str, str, dict[str, Any]] | None:
        if event.event_type not in PERSISTED_EVENT_TYPES:
            return None
        metadata: dict[str, Any] = {
            "run_id": event.run_id,
            "event_sequence": event.sequence,
            "event_type": event.event_type,
        }
        data = event.data
        if event.event_type == "user_message":
            return "user", event.message, {**metadata, "phase": "input"}
        if event.event_type == "run_started":
            return "info", event.message, {
                **metadata,
                "phase": str(data.get("phase", "lifecycle")),
            }
        if event.event_type in {"cancelling", "cancelled"}:
            return "info", event.message, {
                **metadata,
                "phase": event.event_type,
            }
        if event.event_type in {"tool_start", "tool_args", "tool_done"}:
            tool_name = str(data.get("tool_name", ""))
            metadata.update(
                {
                    "tool_name": tool_name,
                    "phase": event.event_type.removeprefix("tool_"),
                }
            )
            if event.event_type == "tool_args":
                content = json.dumps(
                    data.get("arguments", {}), ensure_ascii=False, default=str
                )
            else:
                content = ""
                if event.event_type == "tool_done":
                    metadata.update(
                        {
                            "result_id": str(data.get("result_id", "")),
                            "result_preview": str(data.get("result_preview", "")),
                        }
                    )
            return "tool", content, metadata
        if event.event_type == "final_message":
            return "assistant", event.message, metadata
        if event.event_type == "done":
            return "done", event.message, metadata
        if event.event_type == "error":
            return "error", event.message, metadata
        return None

    async def _append_and_broadcast(self, run: _ChatRun, event: RunEvent) -> None:
        self._assert_can_append(run, event.event_type)
        if event.sequence != run.last_event_sequence + 1:
            raise InvalidRunTransition(
                f"event sequence {event.sequence} does not follow {run.last_event_sequence}"
            )
        if event.terminal:
            if run.status not in TERMINAL_STATUSES:
                raise InvalidRunTransition(
                    f"terminal event {event.event_type} without terminal Run status"
                )
            run.terminal_event_sequence = event.sequence
        elif run.status in TERMINAL_STATUSES:
            raise InvalidRunTransition(
                f"cannot append {event.event_type} after terminal Run status {run.status}"
            )
        run.last_event_sequence = event.sequence
        if event.event_type == "token":
            run.accumulated_draft += event.message
            run.accumulated_draft_through_sequence = event.sequence
        elif event.event_type == "final_message":
            run.accumulated_draft = event.message
            run.accumulated_draft_through_sequence = event.sequence
        elif event.event_type.startswith("tool_"):
            run.recent_tool = dict(event.data)
        run.history_events.append(event)
        for queue in tuple(run.subscribers):
            queue.put_nowait(event)
        await self.event_bus.publish(self._to_agent_event(event))

    @staticmethod
    def _assert_can_append(run: _ChatRun, event_type: str) -> None:
        if run.terminal_event_sequence is not None:
            raise InvalidRunTransition(
                f"cannot append {event_type} after terminal event "
                f"{run.terminal_event_sequence}"
            )

    async def _record_persistence_failure(self, run: _ChatRun, exc: Exception) -> None:
        if run.persistence_failure_reported or run.terminal_event_sequence is not None:
            return
        run.persistence_failure_reported = True
        run.error = f"chat run persistence failed: {exc}"
        run.token.request_cancel()
        if run.status not in TERMINAL_STATUSES:
            self._transition(run, RunStatus.FAILED)
        event = RunEvent(
            sequence=run.last_event_sequence + 1,
            run_id=run.run_id,
            session_id=run.session_id,
            event_type="error",
            message=run.error,
            data={"error": run.error, "persistence_failure": True},
            terminal=True,
        )
        await self._append_and_broadcast(run, event)

    async def _finish_and_maybe_start_successor(self, run: _ChatRun) -> None:
        async with self._session_lock(run.session_id):
            async with self._lock:
                if self._active_by_session.get(run.session_id) != run.run_id:
                    return
                # Keep the successor marker until its Run is registered so a
                # refreshed page/API cannot mistake the handoff window for an
                # idle session or start an unrelated Run.
                successor = self._successors.get(run.session_id)
                self._active_by_session.pop(run.session_id, None)
            if successor is not None:
                try:
                    await self._start_run_locked(
                        run.session_id,
                        successor.message,
                        persist_user_message=successor.persist_user_message,
                        run_id=successor.request_id,
                    )
                except Exception:
                    logger.exception("failed to start successor for %s", run.session_id)
                finally:
                    self._successors.pop(run.session_id, None)
            self._schedule_cleanup(run.run_id)

    def _schedule_cleanup(self, run_id: str) -> None:
        task = asyncio.create_task(
            self._cleanup_after_ttl(run_id),
            name=f"chat-run-cleanup-{run_id}",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _cleanup_after_ttl(self, run_id: str) -> None:
        await asyncio.sleep(self.terminal_ttl_seconds)
        async with self._lock:
            run = self._runs.get(run_id)
            if run is not None and run.status in TERMINAL_STATUSES:
                async with run.event_lock:
                    if run.subscribers:
                        return
                    self._runs.pop(run_id, None)
                    self.event_bus.clear_history(run_id)

    def _transition(self, run: _ChatRun, target: RunStatus) -> None:
        current = run.status
        if current is target:
            return
        if current in TERMINAL_STATUSES:
            raise InvalidRunTransition(f"{current} cannot transition to {target}")
        allowed: dict[RunStatus, set[RunStatus]] = {
            RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.CANCELLING, RunStatus.FAILED},
            RunStatus.RUNNING: {
                RunStatus.CANCELLING,
                RunStatus.CANCELLED,
                RunStatus.COMPLETED,
                RunStatus.FAILED,
            },
            RunStatus.CANCELLING: {
                RunStatus.CANCELLED,
                RunStatus.COMPLETED,
                RunStatus.FAILED,
            },
        }
        if target not in allowed.get(current, set()):
            raise InvalidRunTransition(f"{current} cannot transition to {target}")
        run.status = target
        if target is RunStatus.RUNNING and run.started_at is None:
            run.started_at = shanghai_now_iso()
        if target in TERMINAL_STATUSES:
            run.finished_at = shanghai_now_iso()

    def _snapshot(self, run: _ChatRun | None) -> RunSnapshot:
        if run is None:
            raise KeyError("run is missing")
        successor = self._successors.get(run.session_id)
        return RunSnapshot(
            run_id=run.run_id,
            session_id=run.session_id,
            status=run.status,
            message=run.message,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            error=run.error,
            last_event_sequence=run.last_event_sequence,
            cancellation_requested=run.token.is_cancel_requested(),
            accumulated_draft=run.accumulated_draft,
            accumulated_draft_through_sequence=run.accumulated_draft_through_sequence,
            recent_tool=run.recent_tool,
            successor_run_id=successor.request_id if successor is not None else None,
        )

    @staticmethod
    def _to_agent_event(event: RunEvent) -> AgentEvent:
        mapping: dict[str, AgentEventType] = {
            "run_started": AgentEventType.RUN_STARTED,
            "cancelling": AgentEventType.RUN_CANCELLING,
            "done": AgentEventType.RUN_COMPLETED,
            "cancelled": AgentEventType.RUN_CANCELLED,
            "final_message": AgentEventType.LLM_MESSAGE,
            "token": AgentEventType.LLM_TOKEN_DELTA,
            "tool_start": AgentEventType.TOOL_CALL_STARTED,
            "tool_done": AgentEventType.TOOL_CALL_COMPLETED,
            "todo_status": AgentEventType.TODO_STATUS_UPDATED,
        }
        event_type = mapping.get(event.event_type, AgentEventType.PROGRESS)
        if event.event_type == "error":
            event_type = (
                AgentEventType.RUN_FAILED
                if event.terminal
                else AgentEventType.RUN_DIAGNOSTIC
            )
        experiment_id = event.data.get("experiment_id")
        return AgentEvent(
            run_id=event.run_id,
            experiment_id=str(experiment_id) if experiment_id is not None else None,
            event_type=event_type,
            title=event.event_type,
            message=event.message,
            payload={
                **event.data,
                "sequence": event.sequence,
                "session_id": event.session_id,
                "event_type": event.event_type,
                "terminal": event.terminal,
                **(
                    {"diagnostic": True, "fallback": bool(event.data.get("fallback"))}
                    if event.event_type == "error" and not event.terminal
                    else {}
                ),
            },
        )


def _same_event_marker(message: ChatMessage, event: RunEvent) -> bool:
    metadata = message.metadata
    return (
        metadata.get("run_id") == event.run_id
        and metadata.get("event_sequence") == event.sequence
    )


def _terminal_for_event(event_type: str, data: dict[str, Any]) -> bool:
    """Assign terminal meaning once, at the RunEvent construction boundary."""
    if event_type in {"done", "cancelled"}:
        return True
    return event_type == "error" and not bool(data.get("fallback", False))


def is_terminal_run_event(event: RunEvent) -> bool:
    """Return the explicit terminal meaning shared by all subscribers."""
    return event.terminal
