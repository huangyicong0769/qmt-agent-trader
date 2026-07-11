"""Compatibility facade for shared persistence atomic file operations."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import StorageError
from qmt_agent_trader.persistence.locks import LockManager


def atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path | os.PathLike[str],
    *,
    lock_manager: LockManager,
    assume_locked: bool = False,
) -> None:
    """Preserve the historical function while delegating to the shared store."""
    path = Path(path)
    try:
        store = AtomicFileStore(lock_manager)
        if assume_locked:
            store.write_parquet(path, frame)
        else:
            with lock_manager.resource_lock(path):
                store.write_parquet(path, frame)
    except StorageError as exc:
        if exc.__cause__ is not None:
            raise exc.__cause__ from exc
        raise
