"""Backtests page."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/backtests")
    def backtests_page() -> None:
        shell("Backtests")
        ui.label("Backtests").classes("text-2xl font-semibold")
        root = Path("reports/backtests")
        rows = [
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(root.glob("*"), reverse=True)
            if path.is_file()
        ] if root.exists() else []
        ui.table(columns=_columns(), rows=rows, row_key="path").classes("w-full")


def _columns() -> list[dict[str, str]]:
    return [
        {"name": "name", "label": "Name", "field": "name"},
        {"name": "path", "label": "Path", "field": "path"},
        {"name": "size_bytes", "label": "Bytes", "field": "size_bytes"},
    ]
