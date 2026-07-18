from __future__ import annotations

import asyncio
import threading

import pytest

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import ChatRunManager, RunStatus
from qmt_agent_trader.web.schemas import ChatSession


class _ConcurrentOrchestrator:
    def __init__(self) -> None:
        self.a_release = asyncio.Event()
        self.b_started = asyncio.Event()
        self.b_progressed = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        cancel_requested = kwargs["cancel_requested"]
        assert callable(cancel_requested)
        yield OrchestratorEvent(type="run_started", run_id=run_id, session_id=session_id)
        if session_id == "session_a":
            await self.a_release.wait()
            yield OrchestratorEvent(type="done", run_id=run_id, session_id=session_id)
            return
        self.b_started.set()
        yield OrchestratorEvent(
            type="progress",
            run_id=run_id,
            session_id=session_id,
            message="B is making progress",
        )
        self.b_progressed.set()
        while not cancel_requested():
            await asyncio.sleep(0)
        yield OrchestratorEvent(type="cancelled", run_id=run_id, session_id=session_id)


class _OrderedPersistenceOrchestrator:
    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, session_id=session_id)
        yield OrchestratorEvent(
            type="tool_start",
            run_id=run_id,
            session_id=session_id,
            data={"tool_name": "ordered"},
        )
        yield OrchestratorEvent(type="done", run_id=run_id, session_id=session_id)


@pytest.mark.anyio
async def test_slow_run_a_persistence_does_not_block_run_b_subscription_or_cancellation(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="session_a"))
    repository.create(ChatSession(session_id="session_b"))
    orchestrator = _ConcurrentOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)
    original_persist = manager._persist_event
    a_persist_entered = threading.Event()
    release_a_persist = threading.Event()

    def slow_persist(event) -> None:
        if event.session_id == "session_a" and event.event_type == "user_message":
            a_persist_entered.set()
            assert release_a_persist.wait(timeout=2)
        original_persist(event)

    manager._persist_event = slow_persist
    start_a = asyncio.create_task(manager.start_run("session_a", "slow write"))
    assert await asyncio.to_thread(a_persist_entered.wait, 1)

    started_b = await asyncio.wait_for(
        manager.start_run("session_b", "must stay responsive"),
        timeout=0.5,
    )
    subscription_b = manager.subscribe(started_b.run_id)
    assert (await asyncio.wait_for(anext(subscription_b), timeout=0.5)).event_type == "snapshot"
    await asyncio.wait_for(orchestrator.b_started.wait(), timeout=0.5)
    await asyncio.wait_for(orchestrator.b_progressed.wait(), timeout=0.5)
    cancelling_b = await asyncio.wait_for(
        manager.request_cancel(started_b.run_id),
        timeout=0.5,
    )

    assert cancelling_b is not None
    assert cancelling_b.status is RunStatus.CANCELLING
    release_a_persist.set()
    started_a = await start_a
    orchestrator.a_release.set()
    assert (await manager.wait_for_run(started_a.run_id)).status is RunStatus.COMPLETED
    assert (await manager.wait_for_run(started_b.run_id)).status is RunStatus.CANCELLED
    await subscription_b.aclose()


@pytest.mark.anyio
async def test_one_run_persists_events_in_sequence_even_while_its_first_write_is_blocked(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="ordered_session"))
    manager = ChatRunManager(
        orchestrator=_OrderedPersistenceOrchestrator(),
        repository=repository,
    )
    original_persist = manager._persist_event
    entered_run_started = threading.Event()
    release_run_started = threading.Event()
    persisted_types: list[str] = []

    def ordered_persist(event) -> None:
        persisted_types.append(event.event_type)
        if event.event_type == "run_started":
            entered_run_started.set()
            assert release_run_started.wait(timeout=2)
        original_persist(event)

    manager._persist_event = ordered_persist
    started = await manager.start_run("ordered_session", "ordered")
    assert await asyncio.to_thread(entered_run_started.wait, 1)
    assert persisted_types == ["user_message", "run_started"]

    release_run_started.set()
    assert (await manager.wait_for_run(started.run_id)).status is RunStatus.COMPLETED
    stored = repository.get("ordered_session")
    assert stored is not None
    assert [message.metadata.get("event_type") for message in stored.messages] == [
        "user_message",
        "run_started",
        "tool_start",
        "done",
    ]
