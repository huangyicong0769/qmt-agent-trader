from __future__ import annotations

import pytest

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import ChatRunManager, RunStatus
from qmt_agent_trader.web.event_bus import AgentEventType, EventBus
from qmt_agent_trader.web.schemas import ChatSession


class _FallbackThenCompleteOrchestrator:
    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        yield OrchestratorEvent(
            type="error",
            run_id=run_id,
            session_id=session_id,
            message="fallback diagnostic",
            data={"fallback": True, "error": "stream"},
        )
        yield OrchestratorEvent(
            type="final_message",
            run_id=run_id,
            session_id=session_id,
            message="final answer",
        )
        yield OrchestratorEvent(
            type="done",
            run_id=run_id,
            session_id=session_id,
            message="done",
        )


class _TerminalFailureOrchestrator:
    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        yield OrchestratorEvent(
            type="error",
            run_id=run_id,
            session_id=session_id,
            message="terminal failure",
            data={"error": "fatal"},
        )
        yield OrchestratorEvent(
            type="done",
            run_id=run_id,
            session_id=session_id,
            message="must not publish",
        )


@pytest.mark.anyio
async def test_fallback_error_is_a_diagnostic_event_bus_event_not_run_failure(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="fallback_bus"))
    bus = EventBus()
    manager = ChatRunManager(
        orchestrator=_FallbackThenCompleteOrchestrator(),
        repository=repository,
        bus=bus,
    )

    started = await manager.start_run("fallback_bus", "continue after fallback")
    final = await manager.wait_for_run(started.run_id)
    history = bus.get_history(started.run_id)

    assert final.status is RunStatus.COMPLETED
    assert AgentEventType.RUN_FAILED not in [event.event_type for event in history]
    diagnostic = next(
        event for event in history if event.event_type is AgentEventType.RUN_DIAGNOSTIC
    )
    assert diagnostic.payload["terminal"] is False
    assert diagnostic.payload["fallback"] is True
    assert [event.event_type for event in history].count(AgentEventType.RUN_COMPLETED) == 1


@pytest.mark.anyio
async def test_terminal_error_remains_run_failed_and_never_completes_in_event_bus(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="terminal_bus"))
    bus = EventBus()
    manager = ChatRunManager(
        orchestrator=_TerminalFailureOrchestrator(),
        repository=repository,
        bus=bus,
    )

    started = await manager.start_run("terminal_bus", "fail")
    final = await manager.wait_for_run(started.run_id)
    history = bus.get_history(started.run_id)

    assert final.status is RunStatus.FAILED
    failures = [event for event in history if event.event_type is AgentEventType.RUN_FAILED]
    assert len(failures) == 1
    assert failures[0].payload["terminal"] is True
    assert AgentEventType.RUN_COMPLETED not in [event.event_type for event in history]
    assert all("terminal" in event.payload for event in history)
