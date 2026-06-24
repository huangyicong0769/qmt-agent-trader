"""Artifacts page."""

from __future__ import annotations

from nicegui import ui

from qmt_agent_trader.web.routes.artifacts import list_artifacts
from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/artifacts")
    async def artifacts_page() -> None:
        shell("Artifacts")
        ui.label("Artifacts").classes("text-2xl font-semibold")
        artifacts = await list_artifacts()
        rows = [
            {
                "name": artifact.name,
                "path": artifact.path,
                "size_bytes": artifact.size_bytes,
                "modified_at": artifact.modified_at.isoformat(),
            }
            for artifact in artifacts
        ]
        ui.table(columns=_columns(), rows=rows, row_key="path").classes("w-full")


def _columns() -> list[dict[str, str]]:
    return [
        {"name": "name", "label": "Name", "field": "name"},
        {"name": "path", "label": "Path", "field": "path"},
        {"name": "size_bytes", "label": "Bytes", "field": "size_bytes"},
        {"name": "modified_at", "label": "Modified", "field": "modified_at"},
    ]
