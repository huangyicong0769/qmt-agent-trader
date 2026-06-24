"""Experiments page."""

from __future__ import annotations

from nicegui import ui

from qmt_agent_trader.web.routes.workflows import get_experiment_store
from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/experiments")
    def experiments_page() -> None:
        shell("Experiments")
        ui.label("Experiments").classes("text-2xl font-semibold")
        query = ui.input("Filter").classes("w-full")
        table = ui.table(columns=_columns(), rows=[], row_key="experiment_id").classes("w-full")

        def refresh() -> None:
            records = get_experiment_store().search_experiments(query=query.value or None, limit=50)
            table.rows = [
                {
                    "experiment_id": record.experiment_id,
                    "kind": record.kind,
                    "status": record.status.value,
                    "tags": ", ".join(record.tags),
                    "artifacts": len(record.artifacts),
                }
                for record in records
            ]
            table.update()

        ui.button("Refresh", on_click=refresh).props("color=primary")
        refresh()


def _columns() -> list[dict[str, str]]:
    return [
        {"name": "experiment_id", "label": "Experiment", "field": "experiment_id"},
        {"name": "kind", "label": "Kind", "field": "kind"},
        {"name": "status", "label": "Status", "field": "status"},
        {"name": "tags", "label": "Tags", "field": "tags"},
        {"name": "artifacts", "label": "Artifacts", "field": "artifacts"},
    ]
