from __future__ import annotations

import inspect

import pytest

from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import RunEvent
from qmt_agent_trader.web.schemas import ChatMessage as StoredChatMessage
from qmt_agent_trader.web.schemas import ChatSession
from qmt_agent_trader.web.ui.pages import chat


class _LifecycleClient:
    def __init__(self, *, deleted: bool = False) -> None:
        self.is_deleted = deleted


class _LifecycleTarget:
    def __init__(self, *, deleted: bool = False) -> None:
        self.is_deleted = deleted


def test_deleted_client_drops_new_run_event_without_touching_ui() -> None:
    client = _LifecycleClient(deleted=True)
    transcript = _LifecycleTarget()
    targets = chat._RunRenderTargets(transcript=transcript)
    state = chat._AssistantRenderState()

    chat._render_run_event(
        client,  # type: ignore[arg-type]
        targets,
        RunEvent(
            sequence=1,
            run_id="run_deleted_client",
            session_id="chat_deleted_client",
            event_type="token",
            message="late event",
        ),
        state,
        page_alive=lambda: True,
    )

    assert state.text == ""


def test_deleted_transcript_drops_new_run_event_without_touching_ui() -> None:
    client = _LifecycleClient()
    transcript = _LifecycleTarget(deleted=True)
    targets = chat._RunRenderTargets(transcript=transcript)
    state = chat._AssistantRenderState()

    chat._render_run_event(
        client,  # type: ignore[arg-type]
        targets,
        RunEvent(
            sequence=1,
            run_id="run_deleted_transcript",
            session_id="chat_deleted_transcript",
            event_type="token",
            message="late event",
        ),
        state,
        page_alive=lambda: True,
    )

    assert state.text == ""


def test_snapshot_does_not_duplicate_persisted_final_message(monkeypatch) -> None:
    client = _LifecycleClient()
    transcript = _LifecycleTarget()
    targets = chat._RunRenderTargets(transcript=transcript)
    state = chat._AssistantRenderState(persisted_final=True)
    rendered = []
    monkeypatch.setattr(
        chat,
        "_render_assistant_draft",
        lambda *_args, **_kwargs: rendered.append(True),
    )

    chat._render_run_snapshot(
        client,  # type: ignore[arg-type]
        targets,
        {
            "status": "RUNNING",
            "accumulated_draft": "已经持久化的答案",
        },
        state,
        page_alive=lambda: True,
    )

    assert rendered == []


def test_chat_page_uses_public_lifecycle_guards_and_explicit_transcript_context() -> None:
    source = inspect.getsource(chat)
    assert "client.on_delete" in source
    assert "client.is_deleted" in source
    assert "with transcript:" in source
    assert "card.move" not in source
    assert "move(session.transcript)" not in source
    assert "session.orchestrator.execute_stream" not in source


def test_terminal_semantics_and_sidebar_refresh_are_event_properties() -> None:
    fallback = RunEvent(
        sequence=1,
        run_id="run_fallback",
        session_id="session",
        event_type="error",
        data={"fallback": True},
    )
    done = RunEvent(
        sequence=2,
        run_id="run_fallback",
        session_id="session",
        event_type="done",
        terminal=True,
    )
    token = RunEvent(
        sequence=3,
        run_id="run_fallback",
        session_id="session",
        event_type="token",
        message="A",
    )

    assert fallback.terminal is False
    assert done.terminal is True
    assert chat._should_refresh_sidebar(fallback) is True
    assert chat._should_refresh_sidebar(token) is False
    assert chat._should_refresh_sidebar(done) is True
    assert done.to_dict()["terminal"] is True


def test_successor_subscription_only_follows_the_active_session() -> None:
    assert chat._can_start_successor_subscription(
        "session_a",
        "session_a",
        {"session_a", "session_b"},
        page_alive=True,
    )
    assert not chat._can_start_successor_subscription(
        "session_a",
        "session_b",
        {"session_a", "session_b"},
        page_alive=True,
    )
    assert not chat._can_start_successor_subscription(
        "session_a",
        "session_a",
        {"session_a"},
        page_alive=False,
    )


def test_pending_messages_are_isolated_by_session() -> None:
    pending_by_session: dict[str, list[chat._PendingMessage]] = {}
    pending_a = chat._pending_messages_for(pending_by_session, "session_a")
    pending_b = chat._pending_messages_for(pending_by_session, "session_b")
    pending_a.append(chat._PendingMessage("queue", "A", session_id="session_a"))
    pending_b.append(chat._PendingMessage("guide", "B", session_id="session_b"))

    assert [item.content for item in pending_a] == ["A"]
    assert [item.content for item in pending_b] == ["B"]
    assert pending_a[0].session_id == "session_a"
    assert pending_b[0].session_id == "session_b"
    assert chat._pending_message_session(pending_a[0], "session_b") == "session_a"


@pytest.mark.anyio
async def test_real_nicegui_client_delete_stops_late_run_rendering(tmp_path) -> None:
    from nicegui import ui
    from nicegui.testing import user_simulation

    transcript_holder: list[object] = []

    def root() -> None:
        with ui.column() as transcript:
            transcript_holder.append(transcript)

    async with user_simulation(root=root) as user:
        client = await user.open("/")
        transcript = transcript_holder[0]
        repository = ChatSessionRepository(tmp_path / "sessions")
        repository.create(ChatSession(session_id="session_a", title="A"))
        repository.create(ChatSession(session_id="session_b", title="B"))
        with client:
            session_a = chat._ChatSession(
                name="A",
                sid="session_a",
                repository=repository,
            )
            session_b = chat._ChatSession(
                name="B",
                sid="session_b",
                repository=repository,
            )
        repository.update(
            "session_a",
            lambda current: current.model_copy(
                update={
                    "messages": [
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

        class _IdleManager:
            def get_active_run(self, session_id: str) -> None:
                return None

            def has_pending_successor(self, session_id: str) -> bool:
                return False

        pending_by_session = {
            "session_a": [chat._PendingMessage("queue", "send A", session_id="session_a")],
            "session_b": [chat._PendingMessage("queue", "send B", session_id="session_b")],
        }
        manager = _IdleManager()
        activation = chat._prepare_session_activation(
            session_a,
            repository=repository,
            manager=manager,  # type: ignore[arg-type]
            pending_messages_by_session=pending_by_session,
        )
        assert activation.reloaded is True
        session_a.rebuild_ui(client=client, page_alive=lambda: not client.is_deleted)
        contents = [str(getattr(element, "content", "")) for element in client.elements.values()]
        assert any("final answer" in content for content in contents)
        assert [message.content for message in session_a.messages].count("final answer") == 1
        assert pending_by_session["session_a"][0].ready_to_send is True
        assert pending_by_session["session_b"][0].ready_to_send is False
        element_count_before_resync = len(client.elements)
        assert chat._prepare_session_activation(
            session_a,
            repository=repository,
            manager=manager,  # type: ignore[arg-type]
            pending_messages_by_session=pending_by_session,
        ).reloaded is False
        assert len(client.elements) == element_count_before_resync
        assert session_b.messages == []

        targets = chat._RunRenderTargets(transcript=transcript)
        state = chat._AssistantRenderState()

        chat._render_run_event(
            client,
            targets,
            RunEvent(
                sequence=1,
                run_id="run_real_client",
                session_id="session",
                event_type="token",
                message="before delete",
            ),
            state,
            page_alive=lambda: not client.is_deleted,
        )
        assert state.text == "before delete"

        client.delete()
        element_count_after_delete = len(client.elements)
        late_events = [
            RunEvent(
                sequence=2,
                run_id="run_real_client",
                session_id="session",
                event_type="token",
                message=" late token",
            ),
            RunEvent(
                sequence=3,
                run_id="run_real_client",
                session_id="session",
                event_type="tool_done",
                data={"tool_name": "lookup", "result_preview": "ok"},
            ),
            RunEvent(
                sequence=4,
                run_id="run_real_client",
                session_id="session",
                event_type="error",
                message="late diagnostic",
            ),
            RunEvent(
                sequence=5,
                run_id="run_real_client",
                session_id="session",
                event_type="done",
                terminal=True,
            ),
        ]
        for event in late_events:
            chat._render_run_event(
                client,
                targets,
                event,
                state,
                page_alive=lambda: not client.is_deleted,
            )

        assert client.is_deleted
        assert state.text == "before delete"
        assert len(client.elements) == element_count_after_delete
