"""Settings page."""

from __future__ import annotations

import json

from nicegui import ui

from qmt_agent_trader.web.routes.status import get_config_status, get_data_status
from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/settings")
    async def settings_page() -> None:
        shell("Settings")
        ui.label("Settings").classes("text-2xl font-semibold")
        config = await get_config_status()
        data = await get_data_status()
        with ui.tabs().classes("w-full") as tabs:
            config_tab = ui.tab("Config")
            data_tab = ui.tab("Data")
        with ui.tab_panels(tabs, value=config_tab).classes("w-full"):
            with ui.tab_panel(config_tab):
                ui.code(config.model_dump_json(indent=2), language="json").classes("w-full")
            with ui.tab_panel(data_tab):
                ui.code(json.dumps(data.model_dump(), indent=2), language="json").classes("w-full")
