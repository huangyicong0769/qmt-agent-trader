from __future__ import annotations

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.basic_tools import (
    get_current_time_tool,
    run_shell_command_tool,
    wire,
)
from qmt_agent_trader.core.config import Settings


def test_run_shell_command_allows_read_only_command(tmp_path) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    wire(settings=Settings(project_root=tmp_path))

    result = run_shell_command_tool.run(
        {"argv": ["cat", "sample.txt"]},
        ToolContext(run_id="shell"),
    )

    assert result["status"] == "ok"
    assert result["stdout"] == "hello\n"


def test_run_shell_command_rejects_unlisted_command(tmp_path) -> None:
    wire(settings=Settings(project_root=tmp_path))

    result = run_shell_command_tool.run(
        {"argv": ["python", "-c", "print(1)"]},
        ToolContext(run_id="shell"),
    )

    assert result["status"] == "DENIED"


def test_run_shell_command_rejects_shell_metacharacters(tmp_path) -> None:
    wire(settings=Settings(project_root=tmp_path))

    result = run_shell_command_tool.run(
        {"argv": ["ls", "&&", "pwd"]},
        ToolContext(run_id="shell"),
    )

    assert result["status"] == "DENIED"


def test_run_shell_command_rejects_paths_outside_project(tmp_path) -> None:
    wire(settings=Settings(project_root=tmp_path))

    result = run_shell_command_tool.run(
        {"argv": ["cat", "../secret.txt"]},
        ToolContext(run_id="shell"),
    )

    assert result["status"] == "DENIED"


def test_run_shell_command_times_out(tmp_path) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    wire(settings=Settings(project_root=tmp_path))

    result = run_shell_command_tool.run(
        {"argv": ["tail", "-f", "sample.txt"], "timeout_seconds": 0},
        ToolContext(run_id="shell"),
    )

    assert result["status"] == "TIMEOUT"


def test_run_shell_command_truncates_large_output(tmp_path) -> None:
    (tmp_path / "large.txt").write_text("x" * 25_000, encoding="utf-8")
    wire(settings=Settings(project_root=tmp_path))

    result = run_shell_command_tool.run(
        {"argv": ["cat", "large.txt"]},
        ToolContext(run_id="shell"),
    )

    assert result["status"] == "ok"
    assert result["truncated"] is True
    assert len(result["stdout"].encode("utf-8")) <= 20_000


def test_get_current_time_returns_shanghai_time(tmp_path) -> None:
    wire(settings=Settings(project_root=tmp_path))

    result = get_current_time_tool.run({}, ToolContext(run_id="time"))

    assert result["status"] == "ok"
    assert result["timezone"] == "Asia/Shanghai"
    assert result["date"]
