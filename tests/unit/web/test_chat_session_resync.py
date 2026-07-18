from __future__ import annotations

from dataclasses import dataclass

from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.schemas import ChatMessage as StoredChatMessage
from qmt_agent_trader.web.schemas import ChatSession
from qmt_agent_trader.web.ui.pages import chat


@dataclass
class _SessionViewStub:
    sid: str
    _stored: ChatSession
    name: str
    messages: list[chat.ChatMessage]
    preview: str
    _initial_preview: str
    container: object
    transcript: object


def test_reload_session_from_repository_replaces_stale_messages_without_replacing_targets(
    tmp_path,
) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    initial = repository.create(ChatSession(session_id="session_a", title="stale"))
    canonical = repository.update(
        "session_a",
        lambda current: current.model_copy(
            update={
                "title": "canonical",
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
                            "result_preview": "ok",
                        },
                    ),
                    StoredChatMessage(
                        session_id="session_a",
                        role="assistant",
                        content="后台最终答案",
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
                ],
            }
        ),
    )
    container = object()
    transcript = object()
    session = _SessionViewStub(
        sid="session_a",
        _stored=initial,
        name="stale",
        messages=[],
        preview="old preview",
        _initial_preview="old preview",
        container=container,
        transcript=transcript,
    )

    changed = chat._reload_session_from_repository(session, repository)  # type: ignore[arg-type]

    assert changed is True
    assert session._stored == canonical
    assert session.name == "canonical"
    assert [message.role for message in session.messages] == ["tool", "assistant", "done"]
    assert session.preview == "后台最终答案"
    assert session._initial_preview == "后台最终答案"
    assert session.container is container
    assert session.transcript is transcript
    assert repository.get("session_a") == canonical
    assert chat._reload_session_from_repository(session, repository) is False  # type: ignore[arg-type]


def test_pending_reconciliation_only_readies_messages_when_no_run_or_successor() -> None:
    waiting = [chat._PendingMessage("queue", "send after run", session_id="session_a")]

    assert chat._reconcile_pending_for_session(  # type: ignore[attr-defined]
        waiting,
        has_active_run=True,
        has_pending_successor=False,
    ) == waiting
    assert chat._reconcile_pending_for_session(  # type: ignore[attr-defined]
        waiting,
        has_active_run=False,
        has_pending_successor=True,
    ) == waiting
    ready = chat._reconcile_pending_for_session(  # type: ignore[attr-defined]
        waiting,
        has_active_run=False,
        has_pending_successor=False,
    )
    assert ready[0].ready_to_send is True
    assert waiting[0].ready_to_send is False


def test_reload_deleted_session_is_a_noop_without_rebuilding_page_targets(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    stored = repository.create(ChatSession(session_id="deleted_session", title="before delete"))
    container = object()
    transcript = object()
    session = _SessionViewStub(
        sid="deleted_session",
        _stored=stored,
        name="before delete",
        messages=[],
        preview="before delete",
        _initial_preview="before delete",
        container=container,
        transcript=transcript,
    )
    assert repository.delete("deleted_session") is True

    assert chat._reload_session_from_repository(session, repository) is False  # type: ignore[arg-type]
    assert session.name == "before delete"
    assert session.container is container
    assert session.transcript is transcript
