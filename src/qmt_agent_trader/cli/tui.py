"""Minimal Textual TUI for agent chat and todo status."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.agent.runtime import build_default_runtime
from qmt_agent_trader.cli.todo_render import empty_todo_state, render_todo_panel


class AgentTodoTUI(App[None]):
    CSS = """
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
        todos = self.query_one("#todos", Static)
        log.write(f"[bold]User:[/bold] {prompt}")
        orchestrator = AgentOrchestrator(runtime=build_default_runtime())
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


def run_tui() -> None:
    AgentTodoTUI().run()
