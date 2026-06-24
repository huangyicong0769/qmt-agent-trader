"""Tools page."""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

from qmt_agent_trader.web.routes.tools import get_registry
from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/tools")
    def tools_page() -> None:
        shell("Tools")
        ui.label("Agent Tools").classes("text-2xl font-semibold")
        try:
            tools = get_registry().list_tools()
        except Exception as exc:
            ui.markdown(f"Tool registry unavailable: `{exc}`")
            return

        columns = [
            {"name": "name", "label": "Name", "field": "name", "align": "left"},
            {"name": "permission", "label": "Permission", "field": "permission"},
            {"name": "side_effect_level", "label": "Side Effects", "field": "side_effect_level"},
            {"name": "deterministic", "label": "Deterministic", "field": "deterministic"},
        ]
        ui.table(columns=columns, rows=tools, row_key="name").classes("w-full")
        with ui.card().classes("w-full"):
            ui.label("Manual Run").classes("text-lg font-medium")
            tool_name = ui.input("Tool name").classes("w-full")
            payload = ui.textarea("JSON input", value="{}").classes("w-full")
            result = ui.code("{}", language="json").classes("w-full")

            def run() -> None:
                name = str(tool_name.value or "")
                if not name:
                    ui.notify("Tool name is required.", type="warning")
                    return
                try:
                    input_data = json.loads(str(payload.value or "{}"))
                except json.JSONDecodeError as exc:
                    ui.notify(f"Invalid JSON: {exc}", type="negative")
                    return
                if not isinstance(input_data, dict):
                    ui.notify("JSON input must be an object.", type="negative")
                    return
                outcome: dict[str, Any] = {
                    "status": "display_only",
                    "tool": name,
                    "input": input_data,
                    "message": "Use the /api/tools/{tool_name}/run endpoint for audited execution.",
                }
                result.set_content(json.dumps(outcome, ensure_ascii=False, indent=2))

            ui.button("Run", on_click=run).props("color=primary")
