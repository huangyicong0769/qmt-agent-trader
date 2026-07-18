"""Session activation recovery against canonical chat persistence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import ChatRunManager, RunSnapshot, RunStatus
from qmt_agent_trader.web.schemas import ChatMessage as StoredChatMessage
from qmt_agent_trader.web.schemas import ChatSession
from qmt_agent_trader.web.ui.pages import chat


@dataclass
class _SessionView:
    sid: str
    _stored: ChatSession
    name: str
    messages: list[chat.ChatMessage]
    preview: str
    _initial_preview: str
    container: object
    transcript: object


class _ActivationManager:
    def __init__(self, active: dict[str, RunSnapshot]) -> None:
        self.active = active
        self.successors: set[str] = set()

    def get_active_run(self, session_id: str) -> RunSnapshot | None:
        return self.active.get(session_id)

    def has_pending_successor(self, session_id: str) -> bool:
        return session_id in self.successors


class _BackgroundSessionOrchestrator:
    def __init__(self) -> None:
        self.persisted_tool = asyncio.Event()
        self.release = asyncio.Event()

    async def execute_stream(self, message: str, **kwargs: object):
        run_id = str(kwargs["run_id"])
        session_id = str(kwargs["session_id"])
        yield OrchestratorEvent(type="run_started", run_id=run_id, session_id=session_id)
        yield OrchestratorEvent(
            type="tool_done",
            run_id=run_id,
            session_id=session_id,
            data={
                "tool_name": "lookup",
                "result_id": "result-a",
                "result_preview": "persisted result",
            },
        )
        self.persisted_tool.set()
        await self.release.wait()
        yield OrchestratorEvent(type="token", run_id=run_id, session_id=session_id, message="C")
        yield OrchestratorEvent(
            type="final_message",
            run_id=run_id,
            session_id=session_id,
            message="final answer",
        )
        yield OrchestratorEvent(type="done", run_id=run_id, session_id=session_id, message="done")


def _snapshot(run_id: str, session_id: str) -> RunSnapshot:
    return RunSnapshot(
        run_id=run_id,
        session_id=session_id,
        status=RunStatus.RUNNING,
        message="long task",
        created_at="now",
        started_at="now",
        finished_at=None,
        error=None,
        last_event_sequence=3,
        cancellation_requested=False,
        accumulated_draft="",
        accumulated_draft_through_sequence=0,
        recent_tool=None,
    )


def _view(stored: ChatSession) -> _SessionView:
    return _SessionView(
        sid=stored.session_id,
        _stored=stored,
        name=stored.title,
        messages=[],
        preview="",
        _initial_preview="",
        container=object(),
        transcript=object(),
    )


def test_activation_recovers_background_session_without_disturbing_other_session(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    stored_a = repository.create(ChatSession(session_id="session_a", title="A"))
    stored_b = repository.create(ChatSession(session_id="session_b", title="B"))
    session_a = _view(stored_a)
    session_b = _view(stored_b)
    manager = _ActivationManager({"session_a": _snapshot("run_a", "session_a")})
    pending_by_session = {
        "session_a": [chat._PendingMessage("queue", "send A", session_id="session_a")],
        "session_b": [chat._PendingMessage("queue", "send B", session_id="session_b")],
    }

    repository.update(
        "session_a",
        lambda current: current.model_copy(
            update={
                "messages": [
                    StoredChatMessage(
                        session_id="session_a",
                        role="tool",
                        content="",
                        metadata={
                            "run_id": "run_a",
                            "event_sequence": 3,
                            "event_type": "tool_done",
                            "tool_name": "lookup",
                            "phase": "done",
                            "result_preview": "persisted result",
                        },
                    )
                ]
            }
        ),
    )

    running = chat._prepare_session_activation(  # type: ignore[attr-defined]
        session_a,
        repository=repository,
        manager=manager,  # type: ignore[arg-type]
        pending_messages_by_session=pending_by_session,
    )

    assert running.reloaded is True
    assert running.run_snapshot is not None
    assert running.after_sequence == 3
    assert [message.metadata.get("event_type") for message in session_a.messages] == ["tool_done"]
    assert pending_by_session["session_a"][0].ready_to_send is False
    assert pending_by_session["session_b"][0].ready_to_send is False

    manager.active.clear()
    repository.update(
        "session_a",
        lambda current: current.model_copy(
            update={
                "messages": [
                    *current.messages,
                    StoredChatMessage(
                        session_id="session_a",
                        role="assistant",
                        content="final answer",
                        metadata={
                            "run_id": "run_a",
                            "event_sequence": 4,
                            "event_type": "final_message",
                        },
                    ),
                    StoredChatMessage(
                        session_id="session_a",
                        role="done",
                        content="done",
                        metadata={
                            "run_id": "run_a",
                            "event_sequence": 5,
                            "event_type": "done",
                        },
                    ),
                ]
            }
        ),
    )

    completed = chat._prepare_session_activation(  # type: ignore[attr-defined]
        session_a,
        repository=repository,
        manager=manager,  # type: ignore[arg-type]
        pending_messages_by_session=pending_by_session,
    )

    assert completed.reloaded is True
    assert completed.run_snapshot is None
    assert [message.role for message in session_a.messages] == ["tool", "assistant", "done"]
    assert [message.content for message in session_a.messages].count("final answer") == 1
    assert pending_by_session["session_a"][0].ready_to_send is True
    assert pending_by_session["session_b"][0].ready_to_send is False
    assert chat._prepare_session_activation(  # type: ignore[attr-defined]
        session_a,
        repository=repository,
        manager=manager,  # type: ignore[arg-type]
        pending_messages_by_session=pending_by_session,
    ).reloaded is False
    assert session_b.messages == []


@pytest.mark.anyio
async def test_reentering_running_session_uses_persisted_cursor_then_receives_realtime_events(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    stored_a = repository.create(ChatSession(session_id="session_a", title="A"))
    stored_b = repository.create(ChatSession(session_id="session_b", title="B"))
    session_a = _view(stored_a)
    session_b = _view(stored_b)
    orchestrator = _BackgroundSessionOrchestrator()
    manager = ChatRunManager(orchestrator=orchestrator, repository=repository)
    pending_by_session = {
        "session_a": [chat._PendingMessage("queue", "send A", session_id="session_a")],
        "session_b": [chat._PendingMessage("queue", "send B", session_id="session_b")],
    }

    started = await manager.start_run("session_a", "background task")
    await orchestrator.persisted_tool.wait()
    reentered = chat._prepare_session_activation(
        session_a,
        repository=repository,
        manager=manager,
        pending_messages_by_session=pending_by_session,
    )

    assert reentered.reloaded is True
    assert reentered.run_snapshot is not None
    assert reentered.after_sequence == 3
    assert [message.metadata.get("event_type") for message in session_a.messages] == [
        "user_message",
        "run_started",
        "tool_done",
    ]
    assert pending_by_session["session_a"][0].ready_to_send is False
    subscription = manager.subscribe(started.run_id, after_sequence=reentered.after_sequence)
    assert (await anext(subscription)).event_type == "snapshot"

    orchestrator.release.set()
    resumed = [event async for event in subscription]
    assert [event.event_type for event in resumed] == ["token", "final_message", "done"]
    assert (await manager.wait_for_run(started.run_id)).status is RunStatus.COMPLETED

    completed = chat._prepare_session_activation(
        session_a,
        repository=repository,
        manager=manager,
        pending_messages_by_session=pending_by_session,
    )
    assert completed.reloaded is True
    assert [message.content for message in session_a.messages].count("final answer") == 1
    assert pending_by_session["session_a"][0].ready_to_send is True
    assert pending_by_session["session_b"][0].ready_to_send is False
    assert session_b.messages == []
