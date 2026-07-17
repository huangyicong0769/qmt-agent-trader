from __future__ import annotations

import asyncio

import pytest

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import (
    ChatRunManager,
    RunAlreadyActiveError,
    RunStatus,
    SessionDeletionBlockedError,
    SuccessorAlreadyPendingError,
)
from qmt_agent_trader.web.event_bus import EventBus
from qmt_agent_trader.web.schemas import ChatSession


class _BlockingOrchestrator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        self.started.set()
        yield OrchestratorEvent(
            type="run_started",
            run_id=str(kwargs["run_id"]),
            session_id=str(kwargs["session_id"]),
            message="started",
        )
        await self.release.wait()
        yield OrchestratorEvent(
            type="done",
            run_id=str(kwargs["run_id"]),
            session_id=str(kwargs["session_id"]),
            message="done",
        )


class _CancellableOrchestrator:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        self.started.set()
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        cancel_requested = kwargs["cancel_requested"]
        assert callable(cancel_requested)
        while not cancel_requested():
            await asyncio.sleep(0.001)
        yield OrchestratorEvent(
            type="cancelled",
            run_id=run_id,
            message="cancelled",
            data={"reason": "user_interrupt"},
        )


class _SuccessorOrchestrator:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0
        self.first_started = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        self.calls.append(message)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
            cancel_requested = kwargs["cancel_requested"]
            assert callable(cancel_requested)
            if len(self.calls) == 1:
                self.first_started.set()
                while not cancel_requested():
                    await asyncio.sleep(0.001)
                yield OrchestratorEvent(
                    type="cancelled",
                    run_id=run_id,
                    message="cancelled",
                )
            else:
                yield OrchestratorEvent(type="done", run_id=run_id, message="done")
        finally:
            self.active -= 1


class _ManyEventsOrchestrator:
    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        for index in range(12):
            yield OrchestratorEvent(
                type="progress",
                run_id=run_id,
                message=f"progress {index}",
            )
        yield OrchestratorEvent(type="done", run_id=run_id, message="done")


@pytest.mark.anyio
async def test_start_run_is_manager_owned_and_rejects_overlapping_session_run(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    session = repository.create(ChatSession(session_id="chat_1"))
    orchestrator = _BlockingOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    snapshot = await manager.start_run(session.session_id, "长任务")

    assert snapshot.status is RunStatus.PENDING
    await orchestrator.started.wait()
    active = manager.get_active_run(session.session_id)
    assert active is not None
    assert active.run_id == snapshot.run_id
    assert active.status is RunStatus.RUNNING

    with pytest.raises(RunAlreadyActiveError):
        await manager.start_run(session.session_id, "第二条")

    orchestrator.release.set()
    await manager.wait_for_run(snapshot.run_id)
    assert manager.get_run(snapshot.run_id).status is RunStatus.COMPLETED


@pytest.mark.anyio
async def test_subscriber_receives_snapshot_and_ordered_events_without_owning_run(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    session = repository.create(ChatSession(session_id="chat_2"))
    orchestrator = _BlockingOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    snapshot = await manager.start_run(session.session_id, "订阅测试")
    events = manager.subscribe(snapshot.run_id)
    first = await anext(events)
    assert first.event_type == "snapshot"
    assert first.data["snapshot"]["status"] == RunStatus.PENDING.value

    second = await anext(events)
    assert second.event_type == "user_message"
    assert second.sequence == 1
    await orchestrator.started.wait()
    third = await anext(events)
    assert third.event_type == "run_started"
    assert third.sequence == 2

    await events.aclose()
    assert manager.subscriber_count(snapshot.run_id) == 0
    orchestrator.release.set()
    await manager.wait_for_run(snapshot.run_id)


@pytest.mark.anyio
async def test_multiple_subscribers_receive_one_ordered_event_stream(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_subscribers"))
    bus = EventBus()
    manager = ChatRunManager(
        orchestrator=_BlockingOrchestrator(),
        repository=repository,
        bus=bus,
    )
    orchestrator = manager.orchestrator
    snapshot = await manager.start_run("chat_subscribers", "多订阅者")
    assert isinstance(orchestrator, _BlockingOrchestrator)
    await orchestrator.started.wait()

    first = manager.subscribe(snapshot.run_id)
    second = manager.subscribe(snapshot.run_id)
    first_events = [await anext(first), await anext(first), await anext(first)]
    second_events = [await anext(second), await anext(second), await anext(second)]
    assert [event.sequence for event in first_events] == [0, 1, 2]
    assert [event.sequence for event in second_events] == [0, 1, 2]
    assert manager.subscriber_count(snapshot.run_id) == 2

    await first.aclose()
    assert manager.subscriber_count(snapshot.run_id) == 1
    orchestrator.release.set()
    remaining = [event async for event in second]
    await manager.wait_for_run(snapshot.run_id)
    assert remaining[-1].event_type == "done"
    assert manager.subscriber_count(snapshot.run_id) == 0
    assert [event.payload["sequence"] for event in bus.get_history(snapshot.run_id)] == [
        event.sequence for event in first_events[1:] + remaining
    ]


@pytest.mark.anyio
async def test_cancelled_subscriber_task_is_removed_without_cancelling_run(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_subscriber_cancel"))
    orchestrator = _BlockingOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)
    snapshot = await manager.start_run("chat_subscriber_cancel", "订阅取消")

    async def consume() -> None:
        subscription = manager.subscribe(snapshot.run_id)
        try:
            async for _event in subscription:
                await asyncio.sleep(0)
        finally:
            await subscription.aclose()

    subscriber_task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    assert manager.subscriber_count(snapshot.run_id) == 1
    subscriber_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await subscriber_task
    assert manager.subscriber_count(snapshot.run_id) == 0
    assert manager.get_active_run("chat_subscriber_cancel") is not None

    orchestrator.release.set()
    final = await manager.wait_for_run(snapshot.run_id)
    assert final.status is RunStatus.COMPLETED


@pytest.mark.anyio
async def test_cancel_is_idempotent_and_worker_confirms_cancelled(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_cancel"))
    orchestrator = _CancellableOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    snapshot = await manager.start_run("chat_cancel", "可取消任务")
    await orchestrator.started.wait()
    cancelling = await manager.request_cancel(snapshot.run_id)
    assert cancelling is not None
    assert cancelling.status is RunStatus.CANCELLING
    repeated = await manager.request_cancel(snapshot.run_id)
    assert repeated is not None
    assert repeated.status is RunStatus.CANCELLING

    final = await manager.wait_for_run(snapshot.run_id)
    assert final.status is RunStatus.CANCELLED
    assert manager.get_active_run("chat_cancel") is None
    events = [event async for event in manager.subscribe(snapshot.run_id)]
    event_types = [event.event_type for event in events]
    assert event_types.index("cancelling") < event_types.index("cancelled")


@pytest.mark.anyio
async def test_request_cancel_on_terminal_run_is_idempotent(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_terminal"))
    orchestrator = _BlockingOrchestrator()
    manager = ChatRunManager(
        orchestrator=orchestrator,
        repository=repository,
    )

    started = await manager.start_run("chat_terminal", "完成任务")
    orchestrator.release.set()
    final = await manager.wait_for_run(started.run_id)
    repeated = await manager.request_cancel(started.run_id)

    assert repeated is not None
    assert repeated.status is final.status is RunStatus.COMPLETED


@pytest.mark.anyio
async def test_event_history_is_bounded_and_terminal_runs_expire(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_ttl"))
    bus = EventBus()
    manager = ChatRunManager(
        orchestrator=_ManyEventsOrchestrator(),
        repository=repository,
        bus=bus,
        history_limit=8,
        terminal_ttl_seconds=0.01,
    )

    started = await manager.start_run("chat_ttl", "有限历史")
    await manager.wait_for_run(started.run_id)
    replay = [event async for event in manager.subscribe(started.run_id)]
    assert len([event for event in replay if event.sequence > 0]) <= 8
    assert manager.get_run(started.run_id) is not None

    await asyncio.sleep(0.03)
    assert manager.get_run(started.run_id) is None
    assert bus.get_history(started.run_id) == []


@pytest.mark.anyio
async def test_interrupt_successor_starts_only_after_old_run_is_terminal(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_successor"))
    orchestrator = _SuccessorOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    first = await manager.start_run("chat_successor", "旧消息")
    await orchestrator.first_started.wait()
    old = await manager.interrupt_and_start("chat_successor", "新消息")
    assert old.status is RunStatus.CANCELLING
    assert old.successor_run_id is not None
    with pytest.raises(SuccessorAlreadyPendingError):
        await manager.interrupt_and_start("chat_successor", "重复新消息")

    old_final = await manager.wait_for_run(first.run_id)
    assert old_final.status is RunStatus.CANCELLED
    successor = manager.get_run(old.successor_run_id)
    assert successor is not None
    await manager.wait_for_run(old.successor_run_id)
    assert manager.get_run(old.successor_run_id).status is RunStatus.COMPLETED
    assert orchestrator.calls == ["旧消息", "新消息"]
    assert orchestrator.max_active == 1


@pytest.mark.anyio
async def test_delete_session_is_serialized_with_run_ownership(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_delete_guard"))
    orchestrator = _BlockingOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("chat_delete_guard", "不能删除")
    with pytest.raises(SessionDeletionBlockedError):
        await manager.delete_session("chat_delete_guard")
    assert repository.get("chat_delete_guard") is not None

    orchestrator.release.set()
    await manager.wait_for_run(started.run_id)
    assert await manager.delete_session("chat_delete_guard") is True
    assert repository.get("chat_delete_guard") is None
