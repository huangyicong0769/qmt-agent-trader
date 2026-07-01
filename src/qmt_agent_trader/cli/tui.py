"""Minimal Textual TUI for agent chat and todo status."""

from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.agent.runtime import build_default_runtime
from qmt_agent_trader.cli.todo_render import empty_todo_state, render_todo_panel


def _format_elapsed(total_seconds: int) -> str:
    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _render_run_status(
    *,
    is_running: bool,
    elapsed_seconds: int,
    queue_depth: int = 0,
) -> str:
    if is_running:
        return f"Agent running {_format_elapsed(elapsed_seconds)}"
    if queue_depth:
        return f"Agent idle | queue depth {queue_depth}"
    return "Agent idle"


class AgentTodoTUI(App[None]):
    CSS = """
    #status {
        height: 1;
        color: $success;
    }
    #todos {
        height: 12;
        border: solid $primary;
    }
    #events {
        height: 1fr;
        border: solid $secondary;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(_render_run_status(is_running=False, elapsed_seconds=0), id="status")
        yield Static(render_todo_panel(empty_todo_state()), id="todos")
        yield RichLog(id="events", markup=True, wrap=True)
        yield Input(placeholder="Ask the agent...", id="prompt")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        self.query_one("#prompt", Input).value = ""
        log = self.query_one("#events", RichLog)
        status = self.query_one("#status", Static)
        todos = self.query_one("#todos", Static)
        log.write(f"[bold]User:[/bold] {prompt}")
        orchestrator = AgentOrchestrator(runtime=build_default_runtime())
        started_at = time.monotonic()

        def refresh_status() -> None:
            elapsed = int(time.monotonic() - started_at)
            status.update(
                _render_run_status(is_running=True, elapsed_seconds=elapsed)
            )

        refresh_status()
        timer = self.set_interval(1.0, refresh_status)
        try:
            async for agent_event in orchestrator.execute_stream(
                prompt,
                session_id="tui-default",
            ):
                if agent_event.type == "todo_status":
                    todos.update(render_todo_panel(agent_event.data))
                elif agent_event.type == "token":
                    log.write(agent_event.message)
                elif agent_event.type in {"tool_start", "tool_done", "error", "done"}:
                    log.write(f"[dim]{agent_event.type}[/dim] {agent_event.message}")
        finally:
            timer.stop()
            status.update(_render_run_status(is_running=False, elapsed_seconds=0))


def run_tui() -> None:
    AgentTodoTUI().run()
