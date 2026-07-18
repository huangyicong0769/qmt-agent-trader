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


class _FallbackSequenceOrchestrator:
    def __init__(self, *, fallback: bool) -> None:
        self.fallback = fallback

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        yield OrchestratorEvent(
            type="error",
            run_id=run_id,
            message="diagnostic stream error",
            data={"error": "stream_error", "fallback": self.fallback},
        )
        yield OrchestratorEvent(
            type="final_message",
            run_id=run_id,
            message="最终答案",
            data={"content": "最终答案"},
        )
        yield OrchestratorEvent(type="done", run_id=run_id, message="done")


class _TokenReplayOrchestrator:
    def __init__(self) -> None:
        self.tokens_ready = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        yield OrchestratorEvent(type="token", run_id=run_id, message="A")
        yield OrchestratorEvent(type="token", run_id=run_id, message="B")
        self.tokens_ready.set()
        await self.release.wait()
        yield OrchestratorEvent(type="token", run_id=run_id, message="C")
        yield OrchestratorEvent(type="done", run_id=run_id, message="done")


class _ManyTokenOrchestrator:
    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        for index in range(20):
            yield OrchestratorEvent(type="token", run_id=run_id, message=str(index))
        yield OrchestratorEvent(type="done", run_id=run_id, message="done")


class _ReplayRegistrationRaceOrchestrator:
    def __init__(self, *, terminal_only: bool = False) -> None:
        self.terminal_only = terminal_only
        self.ready = asyncio.Event()
        self.trigger = asyncio.Event()
        self.release = asyncio.Event()

    def prepare(self) -> None:
        self.ready.clear()
        self.trigger.clear()
        self.release.clear()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        self.ready.set()
        await self.trigger.wait()
        if self.terminal_only:
            yield OrchestratorEvent(type="done", run_id=run_id, message="done")
            return
        yield OrchestratorEvent(type="progress", run_id=run_id, message="race")
        await self.release.wait()
        yield OrchestratorEvent(type="done", run_id=run_id, message="done")


class _TerminalGateOrchestrator:
    def __init__(self, event_type: str) -> None:
        self.event_type = event_type
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        self.ready.set()
        await self.release.wait()
        yield OrchestratorEvent(
            type=self.event_type,
            run_id=run_id,
            message=self.event_type,
            data={"error": "failed"} if self.event_type == "error" else {},
        )


class _TerminalPublishBarrierBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.published = asyncio.Event()
        self.release = asyncio.Event()

    async def publish(self, event) -> None:
        await super().publish(event)
        if event.payload["terminal"]:
            self.published.set()
            await self.release.wait()


class _FallbackGateOrchestrator:
    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self.emit_fallback = asyncio.Event()
        self.continue_run = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        self.ready.set()
        await self.emit_fallback.wait()
        yield OrchestratorEvent(
            type="error",
            run_id=run_id,
            message="fallback diagnostic",
            data={"fallback": True, "error": "stream"},
        )
        await self.continue_run.wait()
        yield OrchestratorEvent(type="final_message", run_id=run_id, message="answer")
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
    subscriber_ready = asyncio.Event()

    async def consume() -> None:
        subscription = manager.subscribe(snapshot.run_id)
        try:
            async for _event in subscription:
                subscriber_ready.set()
                await asyncio.sleep(0)
        finally:
            await subscription.aclose()

    subscriber_task = asyncio.create_task(consume())
    await subscriber_ready.wait()
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


@pytest.mark.anyio
async def test_fallback_error_is_diagnostic_and_does_not_end_run(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_fallback"))
    manager = ChatRunManager(
        orchestrator=_FallbackSequenceOrchestrator(fallback=True),
        repository=repository,
    )

    started = await manager.start_run("chat_fallback", "fallback")
    final = await manager.wait_for_run(started.run_id)
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event for event in events if event.sequence > 0]

    assert final.status is RunStatus.COMPLETED
    assert [event.event_type for event in ordered][-3:] == [
        "error",
        "final_message",
        "done",
    ]
    fallback, final_message, done = ordered[-3:]
    assert fallback.terminal is False
    assert final_message.terminal is False
    assert done.terminal is True
    stored = repository.get("chat_fallback")
    assert stored is not None
    assert [message.metadata.get("event_type") for message in stored.messages][-3:] == [
        "error",
        "final_message",
        "done",
    ]


@pytest.mark.anyio
async def test_non_fallback_error_is_terminal_and_blocks_later_events(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_error_terminal"))
    manager = ChatRunManager(
        orchestrator=_FallbackSequenceOrchestrator(fallback=False),
        repository=repository,
    )

    started = await manager.start_run("chat_error_terminal", "error")
    final = await manager.wait_for_run(started.run_id)
    events = [event async for event in manager.subscribe(started.run_id)]
    ordered = [event for event in events if event.sequence > 0]

    assert final.status is RunStatus.FAILED
    assert ordered[-1].event_type == "error"
    assert ordered[-1].terminal is True
    assert all(event.event_type not in {"final_message", "done"} for event in ordered)


@pytest.mark.anyio
async def test_draft_snapshot_filters_already_accumulated_token_replay(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_draft_replay"))
    orchestrator = _TokenReplayOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("chat_draft_replay", "draft")
    await orchestrator.tokens_ready.wait()
    stored = repository.get("chat_draft_replay")
    assert stored is not None
    repository_cursor = max(
        int(message.metadata["event_sequence"])
        for message in stored.messages
        if message.metadata.get("run_id") == started.run_id
    )
    assert repository_cursor == 2

    subscription = manager.subscribe(started.run_id, after_sequence=repository_cursor)
    snapshot_event = await anext(subscription)
    snapshot_data = snapshot_event.data["snapshot"]
    assert snapshot_data["accumulated_draft"] == "AB"
    assert snapshot_data["accumulated_draft_through_sequence"] == 4

    orchestrator.release.set()
    resumed = [event async for event in subscription]
    assert [event.event_type for event in resumed] == ["token", "done"]
    assert resumed[0].message == "C"
    assert manager.get_run(started.run_id).accumulated_draft == "ABC"


@pytest.mark.anyio
async def test_bounded_history_still_recovers_complete_draft_from_snapshot(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_bounded_draft"))
    manager = ChatRunManager(
        orchestrator=_ManyTokenOrchestrator(),
        repository=repository,
        history_limit=8,
    )

    started = await manager.start_run("chat_bounded_draft", "bounded")
    await manager.wait_for_run(started.run_id)
    subscription = manager.subscribe(started.run_id, after_sequence=2)
    snapshot_event = await anext(subscription)
    snapshot_data = snapshot_event.data["snapshot"]

    assert snapshot_data["accumulated_draft"] == "012345678910111213141516171819"
    assert snapshot_data["accumulated_draft_through_sequence"] == 22
    replay = [event async for event in subscription]
    assert [event.event_type for event in replay] == ["done"]


@pytest.mark.anyio
async def test_subscribe_registration_race_delivers_progress_exactly_once_100_times(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_replay_registration_race"))
    orchestrator = _ReplayRegistrationRaceOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    for index in range(100):
        orchestrator.prepare()
        started = await manager.start_run(
            "chat_replay_registration_race",
            f"race-{index}",
        )
        await orchestrator.ready.wait()
        events: list = []
        progress_seen = asyncio.Event()
        barrier = asyncio.Barrier(2)

        async def consume(
            *,
            barrier=barrier,
            run_id=started.run_id,
            events=events,
            progress_seen=progress_seen,
        ) -> None:
            await barrier.wait()
            async for event in manager.subscribe(run_id, after_sequence=2):
                events.append(event)
                if event.event_type == "progress":
                    progress_seen.set()

        async def trigger(*, barrier=barrier) -> None:
            await barrier.wait()
            orchestrator.trigger.set()

        consumer_task = asyncio.create_task(consume())
        await asyncio.gather(trigger(), progress_seen.wait())
        orchestrator.release.set()
        await consumer_task
        await manager.wait_for_run(started.run_id)

        progress_events = [event for event in events if event.event_type == "progress"]
        assert len(progress_events) == 1
        assert [event.sequence for event in events] == sorted(
            event.sequence for event in events
        )


@pytest.mark.anyio
async def test_terminal_registration_race_delivers_done_exactly_once(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_terminal_registration_race"))
    orchestrator = _ReplayRegistrationRaceOrchestrator(terminal_only=True)
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    for index in range(100):
        orchestrator.prepare()
        started = await manager.start_run(
            "chat_terminal_registration_race",
            f"terminal-race-{index}",
        )
        await orchestrator.ready.wait()
        events: list = []
        barrier = asyncio.Barrier(2)

        async def consume(*, barrier=barrier, run_id=started.run_id, events=events) -> None:
            await barrier.wait()
            async for event in manager.subscribe(run_id, after_sequence=2):
                events.append(event)

        async def trigger(*, barrier=barrier) -> None:
            await barrier.wait()
            orchestrator.trigger.set()

        consumer_task = asyncio.create_task(consume())
        await asyncio.gather(trigger(), consumer_task)
        await manager.wait_for_run(started.run_id)
        done_events = [event for event in events if event.event_type == "done"]
        assert len(done_events) == 1
        assert done_events[0].terminal is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("event_type", "expected_status"),
    [
        ("done", RunStatus.COMPLETED),
        ("cancelled", RunStatus.CANCELLED),
        ("error", RunStatus.FAILED),
    ],
)
async def test_terminal_event_is_not_visible_before_terminal_snapshot_commit(
    tmp_path,
    event_type: str,
    expected_status: RunStatus,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id=f"terminal_{event_type}"))
    orchestrator = _TerminalGateOrchestrator(event_type)
    bus = _TerminalPublishBarrierBus()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository, bus=bus)

    started = await manager.start_run(f"terminal_{event_type}", "terminal")
    subscription = manager.subscribe(started.run_id)
    assert (await anext(subscription)).event_type == "snapshot"
    assert (await anext(subscription)).event_type == "user_message"
    await orchestrator.ready.wait()
    assert (await anext(subscription)).event_type == "run_started"

    orchestrator.release.set()
    await bus.published.wait()
    terminal = await anext(subscription)
    visible = manager.get_run(started.run_id)

    assert terminal.terminal is True
    assert visible is not None
    assert visible.status is expected_status
    bus.release.set()
    await subscription.aclose()
    assert (await manager.wait_for_run(started.run_id)).status is expected_status


@pytest.mark.anyio
async def test_fallback_error_keeps_run_nonterminal_until_done(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="fallback_state"))
    orchestrator = _FallbackGateOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)

    started = await manager.start_run("fallback_state", "fallback")
    subscription = manager.subscribe(started.run_id)
    assert (await anext(subscription)).event_type == "snapshot"
    assert (await anext(subscription)).event_type == "user_message"
    await orchestrator.ready.wait()
    assert (await anext(subscription)).event_type == "run_started"

    orchestrator.emit_fallback.set()
    fallback = await anext(subscription)
    visible = manager.get_run(started.run_id)

    assert fallback.event_type == "error"
    assert fallback.terminal is False
    assert visible is not None
    assert visible.status is RunStatus.RUNNING
    orchestrator.continue_run.set()
    remaining = [event async for event in subscription]
    assert [event.event_type for event in remaining] == ["final_message", "done"]
    assert (await manager.wait_for_run(started.run_id)).status is RunStatus.COMPLETED
