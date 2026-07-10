"""Compatibility facade for shared persistence atomic file operations."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import StorageError
from qmt_agent_trader.persistence.locks import LockManager


def atomic_write_parquet(frame: pd.DataFrame, path: Path | os.PathLike[str]) -> None:
    """Preserve the historical function while delegating to the shared store."""
    path = Path(path)
    try:
        AtomicFileStore(LockManager(path.parent / ".locks")).write_parquet(path, frame)
    except StorageError as exc:
        if exc.__cause__ is not None:
            raise exc.__cause__ from exc
        raise
