"""Chat page."""

from __future__ import annotations

from nicegui import ui

from qmt_agent_trader.web.ui.layout import shell


def register() -> None:
    @ui.page("/")
    def chat_page() -> None:
        shell("Chat")
        ui.label("Research Chat").classes("text-2xl font-semibold")
        with ui.row().classes("w-full items-start"):
            with ui.column().classes("w-80"):
                ui.select(
                    ["research", "factor_discovery", "strategy_engineering", "self_bootstrap"],
                    value="research",
                    label="Mode",
                ).classes("w-full")
                ui.input("Theme").classes("w-full")
                ui.input("Universe", value="stock_etf").classes("w-full")
                with ui.row().classes("w-full"):
                    ui.input("Start", value="20200101").classes("grow")
                    ui.input("End", value="20260624").classes("grow")
            with ui.column().classes("grow"):
                transcript = ui.markdown("**Assistant:** Ready for research planning.")
                message = ui.textarea("Message").classes("w-full")

                def send() -> None:
                    content = message.value or ""
                    if not content.strip():
                        ui.notify("Enter a message first.", type="warning")
                        return
                    transcript.set_content(
                        f"**You:** {content}\n\n"
                        "**Assistant:** Agent chat is connected to the Studio shell. "
                        "Full LLM execution is stubbed in v1."
                    )
                    message.value = ""

                ui.button("Send", on_click=send).props("color=primary")
