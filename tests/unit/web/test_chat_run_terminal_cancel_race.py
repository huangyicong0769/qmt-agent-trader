from __future__ import annotations

import asyncio
import threading

import pytest

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import (
    ChatRunManager,
    InvalidRunTransition,
    RunStatus,
)
from qmt_agent_trader.web.event_bus import AgentEventType, EventBus
from qmt_agent_trader.web.schemas import ChatSession


class _TerminalGateOrchestrator:
    def __init__(self, event_type: str) -> None:
        self.event_type = event_type
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, session_id=session_id)
        self.ready.set()
        await self.release.wait()
        yield OrchestratorEvent(
            type=self.event_type,
            run_id=run_id,
            session_id=session_id,
            message=self.event_type,
            data={"error": "failed"} if self.event_type == "error" else {},
        )


class _TerminalCaptureRaceOrchestrator:
    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, session_id=session_id)
        self.ready.set()
        await self.release.wait()
        yield OrchestratorEvent(type="done", run_id=run_id, session_id=session_id)


class _CancelAwareTerminalOrchestrator:
    def __init__(
        self,
        terminal_event_type: str = "done",
        *,
        honor_cancellation: bool = True,
    ) -> None:
        self.terminal_event_type = terminal_event_type
        self.honor_cancellation = honor_cancellation
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        cancel_requested = kwargs["cancel_requested"]
        assert callable(cancel_requested)
        yield OrchestratorEvent(type="run_started", run_id=run_id, session_id=session_id)
        self.ready.set()
        await self.release.wait()
        if self.honor_cancellation and cancel_requested():
            yield OrchestratorEvent(
                type="cancelled",
                run_id=run_id,
                session_id=session_id,
                message="cancelled",
            )
            return
        yield OrchestratorEvent(
            type=self.terminal_event_type,
            run_id=run_id,
            session_id=session_id,
            data={"error": "failed"} if self.terminal_event_type == "error" else {},
        )


class _TeardownGateManager(ChatRunManager):
    """Pause after a terminal event commits but before completion_event is set."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.terminal_committed = asyncio.Event()
        self.release_teardown = asyncio.Event()

    async def _handle_orchestrator_event(self, run, event):  # type: ignore[no-untyped-def]
        terminal = await super()._handle_orchestrator_event(run, event)
        if terminal:
            self.terminal_committed.set()
            await self.release_teardown.wait()
        return terminal


class _ObservableAsyncLock:
    """Expose FIFO waiter arrival while a test owns the Run lock."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.waiter_started = asyncio.Event()
        self.second_waiter_started = asyncio.Event()
        self.waiter_count = 0

    async def __aenter__(self) -> _ObservableAsyncLock:
        if self._lock.locked():
            self.waiter_count += 1
            if self.waiter_count == 1:
                self.waiter_started.set()
            elif self.waiter_count == 2:
                self.second_waiter_started.set()
        await self._lock.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        self._lock.release()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("event_type", "status"),
    [
        ("done", RunStatus.COMPLETED),
        ("cancelled", RunStatus.CANCELLED),
        ("error", RunStatus.FAILED),
    ],
)
async def test_terminal_snapshot_ends_when_cursor_already_covers_terminal_event(
    tmp_path,
    event_type: str,
    status: RunStatus,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id=f"terminal_cursor_{event_type}"))
    orchestrator = _TerminalGateOrchestrator(event_type)
    manager = _TeardownGateManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run(f"terminal_cursor_{event_type}", "terminal")
    await orchestrator.ready.wait()
    orchestrator.release.set()
    await manager.terminal_committed.wait()
    committed = manager.get_run(started.run_id)
    assert committed is not None
    assert committed.status is status
    assert not manager._runs[started.run_id].completion_event.is_set()

    subscription = manager.subscribe(
        started.run_id,
        after_sequence=committed.last_event_sequence,
    )
    try:
        snapshot = await anext(subscription)
        assert snapshot.event_type == "snapshot"
        assert snapshot.data["snapshot"]["status"] == status.value
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(subscription), timeout=0.2)
    finally:
        await subscription.aclose()
        assert manager.subscriber_count(started.run_id) == 0
        manager.release_teardown.set()
        assert (await manager.wait_for_run(started.run_id)).status is status


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("event_type", "status"),
    [
        ("done", RunStatus.COMPLETED),
        ("cancelled", RunStatus.CANCELLED),
        ("error", RunStatus.FAILED),
    ],
)
async def test_terminal_event_replays_once_when_cursor_precedes_it(
    tmp_path,
    event_type: str,
    status: RunStatus,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id=f"terminal_replay_{event_type}"))
    orchestrator = _TerminalGateOrchestrator(event_type)
    manager = _TeardownGateManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run(f"terminal_replay_{event_type}", "terminal")
    await orchestrator.ready.wait()
    orchestrator.release.set()
    await manager.terminal_committed.wait()
    committed = manager.get_run(started.run_id)
    assert committed is not None
    assert committed.status is status

    subscription = manager.subscribe(
        started.run_id,
        after_sequence=committed.last_event_sequence - 1,
    )
    try:
        snapshot = await anext(subscription)
        replayed = await anext(subscription)
        assert snapshot.event_type == "snapshot"
        assert replayed.terminal is True
        assert replayed.sequence == committed.last_event_sequence
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(subscription), timeout=0.2)
    finally:
        await subscription.aclose()
        assert manager.subscriber_count(started.run_id) == 0
        manager.release_teardown.set()
        assert (await manager.wait_for_run(started.run_id)).status is status


@pytest.mark.anyio
async def test_terminal_capture_race_replays_or_queues_done_exactly_once_100_times(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")

    for index in range(100):
        session_id = f"terminal_capture_{index}"
        repository.create(ChatSession(session_id=session_id))
        orchestrator = _TerminalCaptureRaceOrchestrator()
        manager = ChatRunManager(orchestrator=orchestrator, repository=repository)
        started = await manager.start_run(session_id, "terminal capture race")
        await orchestrator.ready.wait()
        barrier = asyncio.Barrier(2)
        received = []

        async def subscribe(
            *,
            barrier=barrier,
            manager=manager,
            run_id=started.run_id,
            received=received,
        ) -> None:
            await barrier.wait()
            async for event in manager.subscribe(run_id, after_sequence=2):
                received.append(event)

        async def commit_terminal(*, barrier=barrier, orchestrator=orchestrator) -> None:
            await barrier.wait()
            orchestrator.release.set()

        await asyncio.gather(subscribe(), commit_terminal())
        assert (await manager.wait_for_run(started.run_id)).status is RunStatus.COMPLETED
        terminal_events = [event for event in received if event.terminal]
        assert len(terminal_events) == 1
        assert terminal_events[0].event_type == "done"
        assert terminal_events[0].sequence == max(
            event.sequence for event in received if event.sequence > 0
        )


@pytest.mark.anyio
async def test_done_commit_wins_over_cancel_waiting_for_the_same_run_lock(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="done_cancel_race"))
    orchestrator = _TerminalGateOrchestrator("done")
    bus = EventBus()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository, bus=bus)
    original_persist = manager._persist_event
    done_persist_entered = threading.Event()
    release_done_persist = threading.Event()

    def block_done_persistence(event) -> None:  # type: ignore[no-untyped-def]
        if event.event_type == "done":
            done_persist_entered.set()
            assert release_done_persist.wait(timeout=2)
        original_persist(event)

    manager._persist_event = block_done_persistence
    started = await manager.start_run("done_cancel_race", "almost done")
    run = manager._runs[started.run_id]
    observable_lock = _ObservableAsyncLock()
    run.event_lock = observable_lock  # type: ignore[assignment]
    await orchestrator.ready.wait()
    orchestrator.release.set()
    assert await asyncio.to_thread(done_persist_entered.wait, 1)

    cancel_task = asyncio.create_task(manager.request_cancel(started.run_id))
    await observable_lock.waiter_started.wait()
    release_done_persist.set()
    cancel_snapshot = await cancel_task
    final = await manager.wait_for_run(started.run_id)

    assert final.status is RunStatus.COMPLETED
    assert cancel_snapshot is not None
    assert cancel_snapshot.status is RunStatus.COMPLETED
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event for event in events if event.sequence > 0]
    assert [event.event_type for event in ordered].count("done") == 1
    assert ordered[-1].event_type == "done"
    assert all(event.event_type != "cancelling" for event in ordered)
    stored = repository.get("done_cancel_race")
    assert stored is not None
    persisted_types = [message.metadata.get("event_type") for message in stored.messages]
    assert persisted_types[-1] == "done"
    assert "cancelling" not in persisted_types
    bus_types = [event.event_type for event in bus.get_history(started.run_id)]
    assert bus_types[-1] is AgentEventType.RUN_COMPLETED
    assert AgentEventType.RUN_CANCELLING not in bus_types


@pytest.mark.anyio
async def test_queued_cancel_converts_already_waiting_done_to_cancelled(tmp_path) -> None:
    """Completion must re-read CANCELLING after its queued lock acquisition."""
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="queued_cancel_before_done_commit"))
    orchestrator = _CancelAwareTerminalOrchestrator(honor_cancellation=False)
    bus = EventBus()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository, bus=bus)

    started = await manager.start_run("queued_cancel_before_done_commit", "race")
    await orchestrator.ready.wait()
    run = manager._runs[started.run_id]
    observable_lock = _ObservableAsyncLock()
    run.event_lock = observable_lock  # type: ignore[assignment]

    async with observable_lock:
        cancel_task = asyncio.create_task(manager.request_cancel(started.run_id))
        await asyncio.wait_for(observable_lock.waiter_started.wait(), timeout=0.2)

        # The completion handler enters while the lock is still held, but it
        # joins the queue after cancellation.  The test therefore exercises
        # the exact stale RUNNING read that used to choose done too early.
        orchestrator.release.set()
        await asyncio.wait_for(observable_lock.second_waiter_started.wait(), timeout=0.2)
        assert observable_lock.waiter_count == 2

    cancel_snapshot = await cancel_task
    final = await manager.wait_for_run(started.run_id)

    assert final.status is RunStatus.CANCELLED
    assert cancel_snapshot is not None
    assert cancel_snapshot.status in {RunStatus.CANCELLING, RunStatus.CANCELLED}

    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event for event in events if event.sequence > 0]
    terminal_events = [event for event in ordered if event.terminal]
    assert [event.event_type for event in ordered][-2:] == ["cancelling", "cancelled"]
    assert all(event.event_type != "done" for event in ordered)
    assert len(terminal_events) == 1
    assert terminal_events[0] is ordered[-1]
    assert terminal_events[0].sequence == max(event.sequence for event in ordered)

    stored = repository.get("queued_cancel_before_done_commit")
    assert stored is not None
    persisted_types = [message.metadata.get("event_type") for message in stored.messages]
    assert persisted_types[-2:] == ["cancelling", "cancelled"]
    assert "done" not in persisted_types

    bus_types = [event.event_type for event in bus.get_history(started.run_id)]
    assert bus_types.count(AgentEventType.RUN_CANCELLING) == 1
    assert bus_types.count(AgentEventType.RUN_CANCELLED) == 1
    assert AgentEventType.RUN_COMPLETED not in bus_types
    assert bus_types[-1] is AgentEventType.RUN_CANCELLED


@pytest.mark.anyio
async def test_cancelling_to_completed_transition_is_rejected(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="reject_cancelling_completion"))
    orchestrator = _CancelAwareTerminalOrchestrator(honor_cancellation=False)
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("reject_cancelling_completion", "stop")
    await orchestrator.ready.wait()
    cancelling = await manager.request_cancel(started.run_id)
    assert cancelling is not None
    assert cancelling.status is RunStatus.CANCELLING

    with pytest.raises(InvalidRunTransition):
        manager._transition(manager._runs[started.run_id], RunStatus.COMPLETED)

    orchestrator.release.set()
    assert (await manager.wait_for_run(started.run_id)).status is RunStatus.CANCELLED


@pytest.mark.anyio
async def test_cancel_commit_precedes_worker_cancelled_terminal(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="cancel_first"))
    orchestrator = _CancelAwareTerminalOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("cancel_first", "stop first")
    await orchestrator.ready.wait()
    cancelling = await manager.request_cancel(started.run_id)
    assert cancelling is not None
    assert cancelling.status is RunStatus.CANCELLING

    orchestrator.release.set()
    final = await manager.wait_for_run(started.run_id)
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event.event_type for event in events if event.sequence > 0]

    assert final.status is RunStatus.CANCELLED
    assert ordered[-2:] == ["cancelling", "cancelled"]
    assert "done" not in ordered


@pytest.mark.anyio
async def test_cancel_commit_wins_over_a_done_event_emitted_after_it(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="cancel_before_late_done"))
    orchestrator = _CancelAwareTerminalOrchestrator(honor_cancellation=False)
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("cancel_before_late_done", "cancel before done")
    await orchestrator.ready.wait()
    cancelling = await manager.request_cancel(started.run_id)
    assert cancelling is not None
    assert cancelling.status is RunStatus.CANCELLING

    orchestrator.release.set()
    final = await manager.wait_for_run(started.run_id)
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event.event_type for event in events if event.sequence > 0]

    assert final.status is RunStatus.CANCELLED
    assert ordered[-2:] == ["cancelling", "cancelled"]
    assert "done" not in ordered


@pytest.mark.anyio
async def test_cancel_commit_precedes_a_pending_terminal_error(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="cancel_before_error"))
    orchestrator = _CancelAwareTerminalOrchestrator("error")
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("cancel_before_error", "cancel before error")
    await orchestrator.ready.wait()
    cancelling = await manager.request_cancel(started.run_id)
    assert cancelling is not None
    assert cancelling.status is RunStatus.CANCELLING

    orchestrator.release.set()
    final = await manager.wait_for_run(started.run_id)
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event.event_type for event in events if event.sequence > 0]

    assert final.status is RunStatus.CANCELLED
    assert ordered[-2:] == ["cancelling", "cancelled"]
    assert "error" not in ordered


@pytest.mark.anyio
async def test_completion_and_cancel_race_keeps_one_terminal_last_event_100_times(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")

    for index in range(100):
        session_id = f"completion_cancel_{index}"
        repository.create(ChatSession(session_id=session_id))
        orchestrator = _CancelAwareTerminalOrchestrator()
        bus = EventBus()
        manager = ChatRunManager(orchestrator=orchestrator, repository=repository, bus=bus)
        started = await manager.start_run(session_id, "completion versus cancel")
        await orchestrator.ready.wait()
        barrier = asyncio.Barrier(2)

        async def request_cancel(
            *,
            barrier=barrier,
            manager=manager,
            run_id=started.run_id,
        ) -> object:
            await barrier.wait()
            return await manager.request_cancel(run_id)

        async def release_completion(*, barrier=barrier, orchestrator=orchestrator) -> None:
            await barrier.wait()
            orchestrator.release.set()

        cancellation, _ = await asyncio.gather(
            request_cancel(),
            release_completion(),
        )
        final = await manager.wait_for_run(started.run_id)
        events = [event async for event in manager.subscribe(started.run_id)]
        ordered = [event for event in events if event.sequence > 0]
        terminal_events = [event for event in ordered if event.terminal]

        assert final.status in {RunStatus.COMPLETED, RunStatus.CANCELLED}
        assert len(terminal_events) == 1
        assert terminal_events[0].sequence == ordered[-1].sequence
        assert all(event.sequence <= terminal_events[0].sequence for event in ordered)
        bus_terminal = [
            event for event in bus.get_history(started.run_id) if event.payload["terminal"]
        ]
        assert len(bus_terminal) == 1
        stored = repository.get(session_id)
        assert stored is not None
        persisted_terminal = [
            message
            for message in stored.messages
            if message.metadata.get("event_type") in {"done", "cancelled", "error"}
        ]
        assert len(persisted_terminal) == 1
        assert cancellation is not None

        types = [event.event_type for event in ordered]
        if final.status is RunStatus.COMPLETED:
            assert types[-1] == "done"
            assert "cancelling" not in types
            assert cancellation.status is RunStatus.COMPLETED
        else:
            assert types[-2:] == ["cancelling", "cancelled"]
            assert cancellation.status in {RunStatus.CANCELLING, RunStatus.CANCELLED}


@pytest.mark.anyio
async def test_terminal_error_commit_wins_over_cancel_waiting_for_the_run_lock(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="error_cancel_race"))
    orchestrator = _TerminalGateOrchestrator("error")
    bus = EventBus()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository, bus=bus)
    original_persist = manager._persist_event
    error_persist_entered = threading.Event()
    release_error_persist = threading.Event()

    def block_error_persistence(event) -> None:  # type: ignore[no-untyped-def]
        if event.event_type == "error":
            error_persist_entered.set()
            assert release_error_persist.wait(timeout=2)
        original_persist(event)

    manager._persist_event = block_error_persistence
    started = await manager.start_run("error_cancel_race", "almost error")
    run = manager._runs[started.run_id]
    observable_lock = _ObservableAsyncLock()
    run.event_lock = observable_lock  # type: ignore[assignment]
    await orchestrator.ready.wait()
    orchestrator.release.set()
    assert await asyncio.to_thread(error_persist_entered.wait, 1)

    cancel_task = asyncio.create_task(manager.request_cancel(started.run_id))
    await observable_lock.waiter_started.wait()
    release_error_persist.set()
    cancel_snapshot = await cancel_task
    final = await manager.wait_for_run(started.run_id)

    assert final.status is RunStatus.FAILED
    assert cancel_snapshot is not None
    assert cancel_snapshot.status is RunStatus.FAILED
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event for event in events if event.sequence > 0]
    assert ordered[-1].event_type == "error"
    assert ordered[-1].terminal is True
    assert all(event.event_type != "cancelling" for event in ordered)
    bus_types = [event.event_type for event in bus.get_history(started.run_id)]
    assert bus_types[-1] is AgentEventType.RUN_FAILED
    assert AgentEventType.RUN_CANCELLING not in bus_types


@pytest.mark.anyio
async def test_terminal_persistence_failure_beats_waiting_cancel_without_late_cancelling(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="failed_done_cancel_race"))
    orchestrator = _TerminalGateOrchestrator("done")
    bus = EventBus()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository, bus=bus)
    original_persist = manager._persist_event
    done_persist_entered = threading.Event()
    release_done_persist = threading.Event()

    def fail_done_persistence(event) -> None:  # type: ignore[no-untyped-def]
        if event.event_type == "done":
            done_persist_entered.set()
            assert release_done_persist.wait(timeout=2)
            raise RuntimeError("terminal persistence failed")
        original_persist(event)

    manager._persist_event = fail_done_persistence
    started = await manager.start_run("failed_done_cancel_race", "persist fail")
    run = manager._runs[started.run_id]
    observable_lock = _ObservableAsyncLock()
    run.event_lock = observable_lock  # type: ignore[assignment]
    await orchestrator.ready.wait()
    orchestrator.release.set()
    assert await asyncio.to_thread(done_persist_entered.wait, 1)

    cancel_task = asyncio.create_task(manager.request_cancel(started.run_id))
    await observable_lock.waiter_started.wait()
    release_done_persist.set()
    cancel_snapshot = await cancel_task
    final = await manager.wait_for_run(started.run_id)

    assert final.status is RunStatus.FAILED
    assert cancel_snapshot is not None
    assert cancel_snapshot.status is RunStatus.FAILED
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event for event in events if event.sequence > 0]
    assert [event.event_type for event in ordered].count("done") == 0
    assert [event.event_type for event in ordered].count("error") == 1
    assert ordered[-1].event_type == "error"
    assert all(event.event_type != "cancelling" for event in ordered)
    stored = repository.get("failed_done_cancel_race")
    assert stored is not None
    persisted_types = [message.metadata.get("event_type") for message in stored.messages]
    assert "done" not in persisted_types
    assert "cancelling" not in persisted_types
    bus_types = [event.event_type for event in bus.get_history(started.run_id)]
    assert bus_types[-1] is AgentEventType.RUN_FAILED
    assert AgentEventType.RUN_CANCELLING not in bus_types


@pytest.mark.anyio
async def test_interrupt_after_terminal_commit_starts_one_successor_without_cancelling(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="terminal_interrupt"))
    orchestrator = _TerminalGateOrchestrator("done")
    manager = _TeardownGateManager(orchestrator=orchestrator, repository=repository)

    first = await manager.start_run("terminal_interrupt", "old message")
    await orchestrator.ready.wait()
    orchestrator.release.set()
    await manager.terminal_committed.wait()

    handoff = await manager.interrupt_and_start("terminal_interrupt", "new message")
    assert handoff.status is RunStatus.COMPLETED
    assert handoff.successor_run_id is not None
    assert manager.has_pending_successor("terminal_interrupt")

    manager.release_teardown.set()
    assert (await manager.wait_for_run(first.run_id)).status is RunStatus.COMPLETED
    successor_id = handoff.successor_run_id
    assert successor_id is not None
    assert (await manager.wait_for_run(successor_id)).status is RunStatus.COMPLETED
    old_events = [event async for event in manager.subscribe(first.run_id)]
    assert all(event.event_type != "cancelling" for event in old_events)
    assert manager.has_pending_successor("terminal_interrupt") is False


@pytest.mark.anyio
async def test_terminal_event_blocks_later_nonterminal_append(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="terminal_append_guard"))
    orchestrator = _TerminalGateOrchestrator("done")
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("terminal_append_guard", "terminal")
    await orchestrator.ready.wait()
    orchestrator.release.set()
    assert (await manager.wait_for_run(started.run_id)).status is RunStatus.COMPLETED
    run = manager._runs[started.run_id]

    with pytest.raises(InvalidRunTransition):
        await manager._emit(run, "progress", "late progress")

    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event for event in events if event.sequence > 0]
    assert ordered[-1].event_type == "done"
