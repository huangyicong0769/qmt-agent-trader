from __future__ import annotations

from qmt_agent_trader.web.ui.pages.chat import _todo_status_markdown


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
