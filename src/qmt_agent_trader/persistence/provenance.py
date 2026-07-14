"""Deterministic fingerprints for file and partitioned-dataset provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def fingerprint_path_tree(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    root = path.parent if path.is_file() else path
    for item in files:
        stat = item.stat()
        digest.update(
            (
                f"{item.relative_to(root).as_posix()}\0"
                f"{stat.st_size}\0{stat.st_mtime_ns}\n"
            ).encode()
        )
    return digest.hexdigest()
