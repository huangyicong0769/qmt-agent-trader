"""Shared NiceGUI layout helpers."""

from __future__ import annotations

from nicegui import ui

NAV_ITEMS = [
    ("Chat", "/"),
    ("Tools", "/tools"),
    ("Runbooks", "/runbooks"),
    ("Experiments", "/experiments"),
    ("Artifacts", "/artifacts"),
    ("Backtests", "/backtests"),
    ("Audit", "/audit"),
    ("Settings", "/settings"),
]


def shell(title: str) -> None:
    with ui.header().classes("items-center justify-between").style("background-color: #1f2937;"):
        ui.label("QMT Agent Studio").classes("text-lg font-semibold")
        ui.label(title).classes("text-sm opacity-80")
    with ui.left_drawer(value=True).classes("bg-grey-2"):
        ui.label("Navigation").classes("text-sm font-semibold q-pa-md")
        for label, target in NAV_ITEMS:
            ui.link(label, target).classes("block q-px-md q-py-sm")
    ui.query(".nicegui-content").classes("q-pa-md")
