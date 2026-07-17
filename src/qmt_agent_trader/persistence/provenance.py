"""Deterministic fingerprints for file and partitioned-dataset provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

_CONTENT_CHUNK_BYTES = 1024 * 1024


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


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
        _update_file(digest, item, root=root)
    return digest.hexdigest()


def _content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CONTENT_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _update_file(
    digest: _Digest,
    item: Path,
    *,
    root: Path,
) -> None:
    stat = item.stat()
    digest.update(
        (
            f"{item.relative_to(root).as_posix()}\0"
            f"{stat.st_size}\0"
            f"{_content_digest(item)}\n"
        ).encode()
    )
