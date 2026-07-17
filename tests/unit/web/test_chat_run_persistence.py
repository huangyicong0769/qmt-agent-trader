from __future__ import annotations

import asyncio

import pytest

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import (
    ChatRunManager,
    RunEvent,
    RunStatus,
)
from qmt_agent_trader.web.event_bus import AgentEventType, EventBus
from qmt_agent_trader.web.schemas import ChatSession


class _ScriptedOrchestrator:
    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        yield OrchestratorEvent(
            type="run_started",
            run_id=run_id,
            session_id=session_id,
            message="started",
        )
        yield OrchestratorEvent(type="token", run_id=run_id, message="草稿")
        yield OrchestratorEvent(
            type="tool_start",
            run_id=run_id,
            message="Calling lookup",
            data={"tool_name": "lookup"},
        )
        yield OrchestratorEvent(
            type="tool_args",
            run_id=run_id,
            message="Args ready",
            data={"tool_name": "lookup", "arguments": {"symbol": "000001.SZ"}},
        )
        yield OrchestratorEvent(
            type="tool_done",
            run_id=run_id,
            message="Tool lookup ✓",
            data={
                "tool_name": "lookup",
                "result_id": "result-1",
                "result_preview": "ok",
            },
        )
        yield OrchestratorEvent(
            type="final_message",
            run_id=run_id,
            message="最终答案",
            data={"content": "最终答案"},
        )
        yield OrchestratorEvent(
            type="done",
            run_id=run_id,
            message="done",
            data={"tool_calls_count": 1},
        )


class _FailureTeardownOrchestrator:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.first_started = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        cancel_requested = kwargs["cancel_requested"]
        assert callable(cancel_requested)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls += 1
        try:
            yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
            if self.calls > 1:
                yield OrchestratorEvent(type="done", run_id=run_id, message="done")
                return
            self.first_started.set()
            while not cancel_requested():
                await asyncio.sleep(0)
            # This event is intentionally the persistence failure point.  The
            # async-generator finally must run before a successor starts.
            yield OrchestratorEvent(
                type="tool_start",
                run_id=run_id,
                message="tool",
                data={"tool_name": "blocking"},
            )
        finally:
            self.active -= 1


class _ToolResultThenCancelledOrchestrator:
    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, message="started")
        yield OrchestratorEvent(
            type="tool_args",
            run_id=run_id,
            message="Args ready",
            data={"tool_name": "lookup", "arguments": {"symbol": "000001.SZ"}},
        )
        yield OrchestratorEvent(
            type="tool_done",
            run_id=run_id,
            message="Tool lookup ✓",
            data={
                "tool_name": "lookup",
                "result_id": "result-cancelled",
                "result_preview": '{"status": "ok"}',
            },
        )
        yield OrchestratorEvent(
            type="cancelled",
            run_id=run_id,
            message="Execution cancelled by user.",
            data={"reason": "user_interrupt"},
        )


@pytest.mark.anyio
async def test_manager_persists_run_events_without_token_deltas(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_persist"))
    manager = ChatRunManager(
        orchestrator=_ScriptedOrchestrator(),
        repository=repository,
    )

    snapshot = await manager.start_run("chat_persist", "执行研究")
    final = await manager.wait_for_run(snapshot.run_id)

    assert final.status is RunStatus.COMPLETED
    stored = repository.get("chat_persist")
    assert stored is not None
    event_types = [message.metadata.get("event_type") for message in stored.messages]
    assert event_types == [
        "user_message",
        "run_started",
        "tool_start",
        "tool_args",
        "tool_done",
        "final_message",
        "done",
    ]
    assert all(message.content != "草稿" for message in stored.messages)
    tool_args = stored.messages[3]
    assert tool_args.metadata["tool_name"] == "lookup"
    assert '"symbol": "000001.SZ"' in tool_args.content
    assert stored.messages[-2].role == "assistant"


@pytest.mark.anyio
async def test_token_events_do_not_schedule_persistence_work(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_token_persist"))
    manager = ChatRunManager(
        orchestrator=_ScriptedOrchestrator(),
        repository=repository,
    )
    original_persist = manager._persist_event
    persisted_types: list[str] = []

    def record_persisted_event(event: RunEvent) -> None:
        persisted_types.append(event.event_type)
        original_persist(event)

    manager._persist_event = record_persisted_event
    started = await manager.start_run("chat_token_persist", "不持久化 token")
    await manager.wait_for_run(started.run_id)

    assert "token" not in persisted_types
    stored = repository.get("chat_token_persist")
    assert stored is not None
    assert all(message.metadata.get("event_type") != "token" for message in stored.messages)


@pytest.mark.anyio
async def test_replay_and_multiple_subscribers_do_not_duplicate_persistence(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_replay"))
    manager = ChatRunManager(
        orchestrator=_ScriptedOrchestrator(),
        repository=repository,
    )

    snapshot = await manager.start_run("chat_replay", "重放")
    await manager.wait_for_run(snapshot.run_id)
    before = repository.get("chat_replay")
    assert before is not None
    before_messages = list(before.messages)

    first = [event async for event in manager.subscribe(snapshot.run_id)]
    second = [event async for event in manager.subscribe(snapshot.run_id, after_sequence=3)]

    after = repository.get("chat_replay")
    assert after is not None
    assert after.messages == before_messages
    assert first[0].event_type == "snapshot"
    assert all(event.sequence > 3 for event in second[1:])
    assert manager.subscriber_count(snapshot.run_id) == 0


@pytest.mark.anyio
async def test_persistence_failure_marks_run_failed_and_emits_one_error(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_persistence_failure"))
    manager = ChatRunManager(
        orchestrator=_ScriptedOrchestrator(),
        repository=repository,
    )
    original_persist = manager._persist_event

    def fail_on_tool_start(event) -> None:
        if event.event_type == "tool_start":
            raise RuntimeError("revision conflict")
        original_persist(event)

    manager._persist_event = fail_on_tool_start
    started = await manager.start_run("chat_persistence_failure", "触发存储失败")
    final = await manager.wait_for_run(started.run_id)
    events = [event async for event in manager.subscribe(started.run_id)]

    assert final.status is RunStatus.FAILED
    errors = [event for event in events if event.event_type == "error"]
    assert len(errors) == 1
    assert errors[0].data["persistence_failure"] is True
    stored = repository.get("chat_persistence_failure")
    assert stored is not None
    assert [message.metadata.get("event_type") for message in stored.messages] == [
        "user_message",
        "run_started",
    ]


@pytest.mark.anyio
async def test_persistence_failure_waits_for_worker_teardown_before_successor(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_failure_successor"))
    orchestrator = _FailureTeardownOrchestrator()
    manager = ChatRunManager(
        orchestrator=orchestrator,
        repository=repository,
    )
    original_persist = manager._persist_event

    def fail_on_tool_start(event) -> None:
        if event.event_type == "tool_start":
            raise RuntimeError("revision conflict")
        original_persist(event)

    manager._persist_event = fail_on_tool_start
    first = await manager.start_run("chat_failure_successor", "旧任务")
    await orchestrator.first_started.wait()
    cancelling = await manager.interrupt_and_start("chat_failure_successor", "新任务")
    assert cancelling.successor_run_id is not None

    old_final = await manager.wait_for_run(first.run_id)
    successor_id = cancelling.successor_run_id
    assert old_final.status is RunStatus.FAILED
    assert successor_id is not None
    successor_final = await manager.wait_for_run(successor_id)

    assert successor_final.status is RunStatus.COMPLETED
    assert orchestrator.active == 0
    assert orchestrator.max_active == 1


@pytest.mark.anyio
async def test_persistence_dedupe_key_is_run_and_sequence_only(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_pair_key"))
    manager = ChatRunManager(
        orchestrator=_ScriptedOrchestrator(),
        repository=repository,
    )
    started = await manager.start_run("chat_pair_key", "幂等")
    await manager.wait_for_run(started.run_id)
    before = repository.get("chat_pair_key")
    assert before is not None

    manager._persist_event(
        RunEvent(
            sequence=2,
            run_id=started.run_id,
            session_id="chat_pair_key",
            event_type="different_type",
            message="must not append",
        )
    )

    after = repository.get("chat_pair_key")
    assert after is not None
    assert after.messages == before.messages


@pytest.mark.anyio
async def test_completed_tool_result_is_persisted_before_cancellation(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(ChatSession(session_id="chat_tool_cancelled"))
    bus = EventBus()
    manager = ChatRunManager(
        orchestrator=_ToolResultThenCancelledOrchestrator(),
        repository=repository,
        bus=bus,
    )

    started = await manager.start_run("chat_tool_cancelled", "停止工具任务")
    final = await manager.wait_for_run(started.run_id)

    assert final.status is RunStatus.CANCELLED
    stored = repository.get("chat_tool_cancelled")
    assert stored is not None
    event_types = [message.metadata.get("event_type") for message in stored.messages]
    assert event_types[-3:] == ["tool_args", "tool_done", "cancelled"]
    assert "final_message" not in event_types
    assert "done" not in event_types
    completed = [
        event
        for event in bus.get_history(started.run_id)
        if event.event_type is AgentEventType.TOOL_CALL_COMPLETED
    ]
    assert len(completed) == 1
    assert completed[0].payload["event_type"] == "tool_done"
