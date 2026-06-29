"""Rich render helpers for agent todo state."""

from __future__ import annotations

from typing import Any, cast

from rich.panel import Panel
from rich.table import Table


def empty_todo_state() -> dict[str, Any]:
    return {
        "summary": {"total": 0, "completed": 0},
        "items": [],
        "active_item": None,
    }


def render_todo_panel(state: dict[str, Any]) -> Panel:
    raw_summary = state.get("summary")
    raw_items = state.get("items")
    summary = cast(dict[str, Any], raw_summary) if isinstance(raw_summary, dict) else {}
    items = raw_items if isinstance(raw_items, list) else []
    total = int(summary.get("total", len(items)))
    completed = int(summary.get("completed", 0))
    title = f"Todo {completed}/{total} completed"

    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    goal = state.get("goal")
    if goal:
        table.add_row(f"Goal: {goal}")
    active = state.get("active_item")
    if isinstance(active, dict):
        table.add_row(f"Active: {active.get('title', '')}")
    if not items:
        table.add_row("No todo items")
        return Panel(table, title=title, border_style="blue")

    item_table = Table(show_header=True, header_style="bold")
    item_table.add_column("Status", no_wrap=True)
    item_table.add_column("Title")
    item_table.add_column("Notes")
    for item in items:
        if not isinstance(item, dict):
            continue
        item_table.add_row(
            str(item.get("status", "PENDING")),
            str(item.get("title", "")),
            str(item.get("notes", "")),
        )
    table.add_row(item_table)
    return Panel(table, title=title, border_style="blue")
