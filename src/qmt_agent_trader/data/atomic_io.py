"""Crash-safe local file writers used by the data lake."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write and validate a Parquet file before atomically replacing its target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temp_path, index=False)
        parquet = pq.ParquetFile(temp_path)  # type: ignore[no-untyped-call]
        for row_group in range(parquet.num_row_groups):
            parquet.read_row_group(row_group)  # type: ignore[no-untyped-call]
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
