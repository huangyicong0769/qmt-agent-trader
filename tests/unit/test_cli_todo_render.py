from __future__ import annotations

from rich.console import Console

from qmt_agent_trader.cli.todo_render import render_todo_panel


def _render_text(state: dict[str, object]) -> str:
    console = Console(record=True, width=100)
    console.print(render_todo_panel(state))
    return console.export_text()


def test_render_todo_panel_handles_empty_state() -> None:
    text = _render_text({"summary": {"total": 0, "completed": 0}, "items": []})

    assert "Todo 0/0 completed" in text
    assert "No todo items" in text


def test_render_todo_panel_shows_active_and_blocked_items() -> None:
    text = _render_text(
        {
            "goal": "研究组合",
            "summary": {"total": 3, "completed": 1},
            "active_item": {"title": "运行回测"},
            "items": [
                {"title": "检查数据", "status": "COMPLETED"},
                {"title": "运行回测", "status": "IN_PROGRESS"},
                {"title": "等待远程数据", "status": "BLOCKED", "notes": "API timeout"},
            ],
        }
    )

    assert "Todo 1/3 completed" in text
    assert "Goal: 研究组合" in text
    assert "Active: 运行回测" in text
    assert "BLOCKED" in text
    assert "API timeout" in text
