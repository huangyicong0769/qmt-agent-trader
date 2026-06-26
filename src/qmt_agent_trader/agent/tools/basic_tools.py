"""Basic read-only utility tools for the research agent."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.config import Settings, get_settings

_settings: Settings | None = None

ALLOWED_COMMANDS = {
    "date",
    "pwd",
    "ls",
    "find",
    "rg",
    "sed",
    "head",
    "tail",
    "wc",
    "cat",
}
FORBIDDEN_TOKENS = {"|", ";", "&&", "||", ">", "<", "`", "$("}
MAX_OUTPUT_BYTES = 20_000


def wire(*, settings: Settings | None = None) -> None:
    global _settings
    _settings = settings or get_settings()


def _get_settings() -> Settings:
    return _settings or get_settings()


def _run_shell_command(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    argv = input_data.get("argv", [])
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        return {"status": "INVALID_REQUEST", "message": "argv must be a non-empty string list"}
    command = argv[0]
    if command not in ALLOWED_COMMANDS:
        return {"status": "DENIED", "message": f"command is not allowed: {command}"}
    denied = _first_forbidden_token(argv)
    if denied is not None:
        return {"status": "DENIED", "message": f"forbidden shell token: {denied}"}
    mutating_arg = _first_mutating_argument(argv)
    if mutating_arg is not None:
        return {
            "status": "DENIED",
            "message": f"mutating option is not allowed for read-only command: {mutating_arg}",
        }

    settings = _get_settings()
    root = settings.project_root.resolve()
    cwd = _resolve_cwd(root, input_data.get("cwd"))
    if cwd is None:
        return {"status": "DENIED", "message": "cwd escapes project root"}
    bad_path = _first_escaping_path(argv[1:], root, cwd)
    if bad_path is not None:
        return {"status": "DENIED", "message": f"path escapes project root: {bad_path}"}

    timeout = int(input_data.get("timeout_seconds", 30))
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "timeout_seconds": timeout, "argv": argv}

    stdout, stdout_truncated = _truncate(completed.stdout)
    stderr, stderr_truncated = _truncate(completed.stderr)
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "argv": argv,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": stdout_truncated or stderr_truncated,
    }


def _get_current_time(_input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    return {
        "status": "ok",
        "timezone": "Asia/Shanghai",
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
    }


def _resolve_cwd(root: Path, raw_cwd: object) -> Path | None:
    if raw_cwd in (None, ""):
        return root
    candidate = (root / str(raw_cwd)).resolve()
    return candidate if _is_under(candidate, root) else None


def _first_forbidden_token(argv: list[str]) -> str | None:
    for arg in argv:
        for token in FORBIDDEN_TOKENS:
            if token in arg:
                return token
    return None


def _first_escaping_path(args: list[str], root: Path, cwd: Path) -> str | None:
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-e", "-m", "-n"}:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        candidate = Path(arg)
        if not candidate.is_absolute() and ".." not in candidate.parts:
            continue
        resolved = (
            (cwd / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        if not _is_under(resolved, root):
            return arg
    return None


def _first_mutating_argument(argv: list[str]) -> str | None:
    command = argv[0] if argv else ""
    if command == "sed":
        for arg in argv[1:]:
            if arg == "-i" or arg.startswith("-i"):
                return arg
    if command == "find":
        mutating_actions = {
            "-delete",
            "-exec",
            "-execdir",
            "-ok",
            "-okdir",
            "-fprint",
            "-fprintf",
            "-fls",
        }
        for arg in argv[1:]:
            if arg in mutating_actions:
                return arg
    return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _truncate(value: str) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_OUTPUT_BYTES:
        return value, False
    return encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore"), True


run_shell_command_tool: AgentTool = tool(
    ToolSpec(
        name="run_shell_command",
        description="运行受控只读 CLI 命令，仅允许安全 allowlist。",
        permission=PermissionLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "integer"},
            },
            "required": ["argv"],
        },
        timeout_seconds=30,
        deterministic=False,
        llm_callable=False,
    ),
    fn=_run_shell_command,
)

get_current_time_tool: AgentTool = tool(
    ToolSpec(
        name="get_current_time",
        description="返回当前 Asia/Shanghai 日期和时间。",
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_get_current_time,
)
