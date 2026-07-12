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
    "core/config.py": {"cwd_relative_persistence_root"},
    "web/config.py": {"cwd_relative_persistence_root"},
}


def scan_forbidden_persistence(root: Path) -> list[PersistenceViolation]:
    violations: list[PersistenceViolation] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except OSError:
            violations.append(PersistenceViolation(path, 0, "unreadable_python"))
            continue
        except SyntaxError as exc:
            violations.append(PersistenceViolation(path, exc.lineno or 0, "invalid_python"))
            continue
        duckdb_connect_aliases = {
            alias.asname or alias.name
            for item in tree.body
            if isinstance(item, ast.ImportFrom) and item.module == "duckdb"
            for alias in item.names
            if alias.name == "connect"
        }
        duckdb_module_aliases = {
            alias.asname or alias.name
            for item in tree.body
            if isinstance(item, ast.Import)
            for alias in item.names
            if alias.name == "duckdb"
        }
        relative = path.relative_to(root).as_posix()
        if root.name == "qmt_agent_trader":
            relative = relative
        allowed = _ALLOWLIST.get(relative, set())
        for node in ast.walk(tree):
            primitive = _primitive(node, duckdb_connect_aliases, duckdb_module_aliases)
            if primitive is not None and primitive not in allowed:
                violations.append(
                    PersistenceViolation(path, int(getattr(node, "lineno", 0)), primitive)
                )
    return violations


def _primitive(
    node: ast.AST,
    duckdb_connect_aliases: set[str],
    duckdb_module_aliases: set[str],
) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if isinstance(function, ast.Name) and function.id == "Path" and node.args:
        value = node.args[0]
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.split("/", 1)[0] in {"reports", "data", "sessions", "approvals"}
        ):
            return "cwd_relative_persistence_root"
    if isinstance(function, ast.Attribute) and function.attr == "to_parquet":
        return "DataFrame.to_parquet"
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "connect"
        and isinstance(function.value, ast.Name)
        and function.value.id in duckdb_module_aliases
    ):
        return "duckdb.connect"
    if isinstance(function, ast.Name) and function.id in duckdb_connect_aliases:
        return "duckdb.connect"
    if isinstance(function, ast.Name) and function.id == "open":
        mode = _open_mode(node)
        if mode is not None and "a" in mode:
            return 'open(..., "a")'
    if isinstance(function, ast.Attribute) and function.attr == "open":
        mode = _open_mode(node)
        if mode is not None and "a" in mode:
            return 'Path.open(..., "a")'
    return None


def _open_mode(node: ast.Call) -> str | None:
    value: ast.AST | None = node.args[1] if len(node.args) >= 2 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            value = keyword.value
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
