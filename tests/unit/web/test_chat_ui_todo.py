from __future__ import annotations

import pytest

from qmt_agent_trader.persistence.errors import StorageRevisionConflictError
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.schemas import ChatMessage as StoredChatMessage
from qmt_agent_trader.web.schemas import ChatSession as StoredChatSession
from qmt_agent_trader.web.ui.pages.chat import (
    _build_pending_message,
    _ChatSession,
    _format_elapsed,
    _mark_pending_ready,
    _pending_message_status,
    _todo_status_markdown,
)


def test_ui_load_save_roundtrip_preserves_canonical_session_fields(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    original = repository.create(
        StoredChatSession(
            session_id="s42",
            title="Canonical",
            created_at="2026-01-01T00:00:00+08:00",
            updated_at="2026-01-02T00:00:00+08:00",
            context={"api": {"keep": True}, "legacy_ui": {"counter": 42, "preview": "p"}},
            messages=[StoredChatMessage(
                message_id="msg_keep", session_id="s42", role="user", content="hello",
                created_at="2026-01-01T01:00:00+08:00", metadata={"keep": 1})],
        )
    )
    ui_session = _ChatSession.load_all(repository)[0]
    ui_session.save()
    saved = repository.get("s42")
    assert saved is not None
    assert saved.created_at == original.created_at
    assert saved.updated_at == original.updated_at
    assert saved.context["api"] == {"keep": True}
    assert saved.messages[0].message_id == "msg_keep"
    assert saved.messages[0].session_id == "s42"
    assert saved.messages[0].created_at == "2026-01-01T01:00:00+08:00"
    assert saved.revision == original.revision == 1


def test_ui_stale_save_raises_visible_revision_conflict(tmp_path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    repository.create(StoredChatSession(session_id="s1", title="one"))
    ui_session = _ChatSession.load_all(repository)[0]
    repository.update("s1", lambda session: session.model_copy(update={"title": "external"}))
    ui_session.name = "ui edit"
    with pytest.raises(StorageRevisionConflictError):
        ui_session.save()


def test_todo_status_markdown_renders_empty_state() -> None:
    rendered = _todo_status_markdown(
        {"summary": {"total": 0, "completed": 0}, "items": []}
    )

    assert "**Todo** 0/0 completed" in rendered
    assert "No todo items." in rendered


def test_todo_status_markdown_renders_active_and_blocked_items() -> None:
    rendered = _todo_status_markdown(
        {
            "goal": "研究组合",
            "summary": {"total": 2, "completed": 0},
            "active_item": {"title": "检查数据"},
            "items": [
                {"title": "检查数据", "status": "IN_PROGRESS"},
                {"title": "等待远程数据", "status": "BLOCKED", "notes": "API timeout"},
            ],
        }
    )

    assert "Goal: 研究组合" in rendered
    assert "Active: 检查数据" in rendered
    assert "`IN_PROGRESS` 检查数据" in rendered
    assert "`BLOCKED` 等待远程数据 - API timeout" in rendered


def test_pending_queue_message_waits_for_confirmation_before_send() -> None:
    pending = _build_pending_message("queue", "  当前任务后检查数据覆盖  ")
    assert pending is not None

    waiting = _pending_message_status(pending)

    assert waiting.badge == "排队中"
    assert "等待当前任务完成" in waiting.detail
    assert "自动执行" not in waiting.detail
    assert waiting.can_undo is True
    assert waiting.can_send is False

    ready = _pending_message_status(_mark_pending_ready(pending))

    assert ready.badge == "排队待确认"
    assert "发送前可撤销" in ready.detail
    assert ready.can_undo is True
    assert ready.can_send is True


def test_pending_queue_status_includes_depth() -> None:
    pending = _build_pending_message("queue", "当前任务后检查数据覆盖")
    assert pending is not None

    waiting = _pending_message_status(pending, queue_depth=2)
    ready = _pending_message_status(_mark_pending_ready(pending), queue_depth=1)

    assert waiting.badge == "排队中 · 2"
    assert "队列深度 2" in waiting.detail
    assert ready.badge == "排队待确认 · 1"
    assert "队列深度 1" in ready.detail


def test_pending_guide_message_has_distinct_visible_status() -> None:
    pending = _build_pending_message("guide", "补充用全市场ETF一起验证")
    assert pending is not None

    waiting = _pending_message_status(pending)
    ready = _pending_message_status(_mark_pending_ready(pending))

    assert waiting.badge == "引导中"
    assert "引导消息" in waiting.detail
    assert ready.badge == "引导待确认"
    assert ready.can_send is True


def test_empty_pending_message_is_ignored() -> None:
    assert _build_pending_message("queue", "   ") is None


def test_format_elapsed_uses_compact_clock_units() -> None:
    assert _format_elapsed(0) == "00:00"
    assert _format_elapsed(65) == "01:05"
    assert _format_elapsed(3661) == "01:01:01"
