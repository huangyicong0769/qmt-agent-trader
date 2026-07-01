from __future__ import annotations

from qmt_agent_trader.cli.tui import _format_elapsed, _render_run_status


def test_tui_run_status_includes_running_elapsed_time() -> None:
    assert _render_run_status(is_running=True, elapsed_seconds=65) == "Agent running 01:05"


def test_tui_run_status_includes_idle_queue_depth() -> None:
    assert _render_run_status(is_running=False, elapsed_seconds=0, queue_depth=2) == (
        "Agent idle | queue depth 2"
    )


def test_tui_format_elapsed_uses_hours_when_needed() -> None:
    assert _format_elapsed(3661) == "01:01:01"
