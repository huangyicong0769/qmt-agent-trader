from __future__ import annotations

import inspect

from qmt_agent_trader.web.chat_run_manager import RunEvent
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
