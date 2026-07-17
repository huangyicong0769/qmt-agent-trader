"""In-process lifecycle checks for application-owned chat runs."""

from __future__ import annotations

import asyncio

import pytest

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import ChatRunManager, RunStatus
from qmt_agent_trader.web.schemas import ChatSession


class _LongRunOrchestrator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        cancel_requested = kwargs["cancel_requested"]
        assert callable(cancel_requested)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            yield OrchestratorEvent(
                type="run_started",
                run_id=run_id,
                session_id=session_id,
                message="started",
            )
            while not self.release.is_set():
                if cancel_requested():
                    yield OrchestratorEvent(
                        type="cancelled",
                        run_id=run_id,
                        session_id=session_id,
                        message="cancelled",
                    )
                    return
                await asyncio.sleep(0)
            yield OrchestratorEvent(
                type="final_message",
                run_id=run_id,
                session_id=session_id,
                message="完成结果",
            )
            yield OrchestratorEvent(
                type="done",
                run_id=run_id,
                session_id=session_id,
                message="done",
            )
        finally:
            self.active -= 1


class _ReplayRaceOrchestrator:
    def __init__(self) -> None:
        self.replay_ready = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        yield OrchestratorEvent(type="progress", run_id=run_id, message="replay me")
        self.replay_ready.set()
        await self.release.wait()
        yield OrchestratorEvent(type="done", run_id=run_id, message="done")


@pytest.mark.anyio
async def test_unsubscribing_page_does_not_cancel_run_or_duplicate_storage(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_lifecycle"))
    orchestrator = _LongRunOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("chat_lifecycle", "页面断开后继续")
    subscription = manager.subscribe(started.run_id)
    assert (await anext(subscription)).event_type == "snapshot"
    assert (await anext(subscription)).event_type == "user_message"
    await orchestrator.started.wait()
    await subscription.aclose()

    assert manager.subscriber_count(started.run_id) == 0
    assert manager.get_run(started.run_id).status is RunStatus.RUNNING
    orchestrator.release.set()
    final = await manager.wait_for_run(started.run_id)

    assert final.status is RunStatus.COMPLETED
    assert orchestrator.active == 0
    stored = repository.get("chat_lifecycle")
    assert stored is not None
    markers = [
        (message.metadata.get("run_id"), message.metadata.get("event_sequence"))
        for message in stored.messages
    ]
    assert len(markers) == len(set(markers))
    assert [message.metadata.get("event_type") for message in stored.messages][-2:] == [
        "final_message",
        "done",
    ]


@pytest.mark.anyio
async def test_new_subscription_recovers_active_run_after_disconnect(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_refresh"))
    orchestrator = _LongRunOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("chat_refresh", "刷新后恢复")
    first = manager.subscribe(started.run_id)
    snapshot_event = await anext(first)
    assert snapshot_event.data["snapshot"]["status"] == RunStatus.PENDING.value
    await anext(first)
    await orchestrator.started.wait()
    run_snapshot = manager.get_run(started.run_id)
    assert run_snapshot is not None
    cursor = run_snapshot.last_event_sequence
    await first.aclose()

    second = manager.subscribe(started.run_id, after_sequence=cursor)
    recovered_snapshot = await anext(second)
    assert recovered_snapshot.sequence == 0
    assert recovered_snapshot.data["snapshot"]["status"] == RunStatus.RUNNING.value

    orchestrator.release.set()
    resumed = [event async for event in second]
    assert [event.event_type for event in resumed][-2:] == ["final_message", "done"]
    assert all(event.sequence > cursor for event in resumed)
    final = await manager.wait_for_run(started.run_id)
    assert final.status is RunStatus.COMPLETED


@pytest.mark.anyio
async def test_subscription_does_not_drop_done_when_completion_races_replay(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_replay_race"))
    orchestrator = _ReplayRaceOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("chat_replay_race", "重放竞态")
    await orchestrator.replay_ready.wait()

    subscription = manager.subscribe(started.run_id, after_sequence=1)
    assert (await anext(subscription)).event_type == "snapshot"
    orchestrator.release.set()
    await manager.wait_for_run(started.run_id)
    replayed = [event async for event in subscription]

    assert [event.event_type for event in replayed][-1] == "done"
    assert [event.sequence for event in replayed] == sorted(
        event.sequence for event in replayed
    )
    assert (await manager.wait_for_run(started.run_id)).status is RunStatus.COMPLETED
