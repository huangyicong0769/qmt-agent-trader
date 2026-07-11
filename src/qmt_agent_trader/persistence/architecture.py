"""AST enforcement for persistence primitives owned by infrastructure."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersistenceViolation:
    path: Path
    line: int
    primitive: str


_ALLOWLIST = {
    "persistence/atomic_files.py": {"DataFrame.to_parquet"},
    "persistence/database.py": {"duckdb.connect"},
}


def scan_forbidden_persistence(root: Path) -> list[PersistenceViolation]:
    violations: list[PersistenceViolation] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        relative = path.relative_to(root).as_posix()
        if root.name == "qmt_agent_trader":
            relative = relative
        allowed = _ALLOWLIST.get(relative, set())
        for node in ast.walk(tree):
            primitive = _primitive(node)
            if primitive is not None and primitive not in allowed:
                violations.append(
                    PersistenceViolation(path, int(getattr(node, "lineno", 0)), primitive)
                )
    return violations


def _primitive(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if isinstance(function, ast.Attribute) and function.attr == "write_text":
        if (
            isinstance(function.value, ast.Call)
            and isinstance(function.value.func, ast.Name)
            and function.value.func.id == "AtomicFileStore"
        ):
            return None
        return "Path.write_text"
    if isinstance(function, ast.Attribute) and function.attr == "to_parquet":
        return "DataFrame.to_parquet"
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "connect"
        and isinstance(function.value, ast.Name)
        and function.value.id == "duckdb"
    ):
        return "duckdb.connect"
    if isinstance(function, ast.Name) and function.id == "open" and len(node.args) >= 2:
        mode = node.args[1]
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "a" in mode.value:
            return 'open(..., "a")'
    return None
