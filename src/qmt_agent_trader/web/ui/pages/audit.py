"""Audit page."""

from __future__ import annotations

from nicegui import ui

from qmt_agent_trader.web.routes.audit import list_tool_calls
from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/audit")
    async def audit_page() -> None:
        shell("Audit")
        ui.label("Audit").classes("text-2xl font-semibold")
        entries = await list_tool_calls(limit=100)
        rows = [
            {
                "timestamp": entry.timestamp,
                "tool_name": entry.tool_name,
                "permission": entry.permission,
                "status": entry.status,
                "duration_ms": entry.duration_ms,
            }
            for entry in entries
        ]
        ui.table(columns=_columns(), rows=rows, row_key="timestamp").classes("w-full")


def _columns() -> list[dict[str, str]]:
    return [
        {"name": "timestamp", "label": "Time", "field": "timestamp"},
        {"name": "tool_name", "label": "Tool", "field": "tool_name"},
        {"name": "permission", "label": "Permission", "field": "permission"},
        {"name": "status", "label": "Status", "field": "status"},
        {"name": "duration_ms", "label": "Duration", "field": "duration_ms"},
    ]
