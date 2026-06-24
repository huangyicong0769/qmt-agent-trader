"""NiceGUI page registration."""

from __future__ import annotations

from qmt_agent_trader.web.ui.pages import (
    artifacts,
    audit,
    backtests,
    chat,
    experiments,
    settings,
    tools,
    workflows,
)

_registered = False


def create_ui() -> None:
    global _registered
    if _registered:
        return
    chat.register()
    tools.register()
    workflows.register()
    experiments.register()
    artifacts.register()
    backtests.register()
    audit.register()
    settings.register()
    _registered = True
