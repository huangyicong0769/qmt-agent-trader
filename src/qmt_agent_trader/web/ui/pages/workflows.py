"""Runbook page — manual workflow templates for advanced users."""

from __future__ import annotations

import json

from nicegui import ui

from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/runbooks")
    def runbooks_page() -> None:
        shell("Runbooks")
        ui.label("Runbooks").classes("text-2xl font-semibold")
        ui.label(
            "Fixed workflow templates for advanced users. "
            "Prefer natural conversation from the Chat page instead."
        ).classes("text-sm text-gray-500 mb-4")
        with ui.tabs().classes("w-full") as tabs:
            factor_tab = ui.tab("Factor Discovery")
            strategy_tab = ui.tab("Strategy Engineering")
            bootstrap_tab = ui.tab("Self Bootstrap")
        with ui.tab_panels(tabs, value=factor_tab).classes("w-full"):
            with ui.tab_panel(factor_tab):
                _factor_panel()
            with ui.tab_panel(strategy_tab):
                _strategy_panel()
            with ui.tab_panel(bootstrap_tab):
                _bootstrap_panel()


def _factor_panel() -> None:
    theme = ui.input("Theme").classes("w-full")
    universe = ui.input("Universe", value="stock_etf").classes("w-full")
    start = ui.input("Start", value="20200101")
    end = ui.input("End", value="20260624")
    output = ui.code("{}", language="json").classes("w-full")

    def run() -> None:
        payload = {
            "workflow": "factor_discovery",
            "theme": theme.value,
            "universe": universe.value,
            "start_date": start.value,
            "end_date": end.value,
        }
        output.set_content(json.dumps(payload, ensure_ascii=False, indent=2))
        ui.notify("Workflow payload prepared. Submit through the API to execute.")

    ui.button("Run Factor Discovery", on_click=run).props("color=primary")


def _strategy_panel() -> None:
    idea = ui.textarea("Strategy idea").classes("w-full")
    factors = ui.input("Factors (comma-separated)").classes("w-full")
    output = ui.code("{}", language="json").classes("w-full")

    def run() -> None:
        selected = [item.strip() for item in str(factors.value or "").split(",") if item.strip()]
        payload = {
            "workflow": "strategy_engineering",
            "strategy_idea": idea.value,
            "selected_factors": selected,
        }
        output.set_content(json.dumps(payload, ensure_ascii=False, indent=2))
        ui.notify("Workflow payload prepared. Submit through the API to execute.")

    ui.button("Run Strategy Engineering", on_click=run).props("color=primary")


def _bootstrap_panel() -> None:
    experiments = ui.input("Experiment IDs (comma-separated)").classes("w-full")
    output = ui.code("{}", language="json").classes("w-full")

    def run() -> None:
        selected = [
            item.strip() for item in str(experiments.value or "").split(",") if item.strip()
        ]
        payload = {"workflow": "self_bootstrap", "recent_experiment_ids": selected}
        output.set_content(json.dumps(payload, ensure_ascii=False, indent=2))
        ui.notify("Workflow payload prepared. Submit through the API to execute.")

    ui.button("Run Self Bootstrap", on_click=run).props("color=primary")
