"""DuckDB + Parquet data lake helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class DataLake:
    def __init__(self, root: Path, duckdb_path: Path) -> None:
        self.root = root
        self.duckdb_path = duckdb_path
        self.root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.duckdb_path))

    def dataset_path(self, layer: str, name: str) -> Path:
        return self.root / layer / f"{name}.parquet"

    def write_parquet(self, frame: pd.DataFrame, layer: str, name: str) -> Path:
        path = self.dataset_path(layer, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = frame
        if len(frame.columns) == 0:
            writable = pd.DataFrame({"_empty": pd.Series(dtype="bool")})
        writable.to_parquet(path, index=False)
        return path

    def read_parquet(self, layer: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.dataset_path(layer, name))

    def list_dataset_names(self, layer: str, prefix: str | None = None) -> list[str]:
        layer_dir = self.root / layer
        if not layer_dir.exists():
            return []
        names = sorted(path.stem for path in layer_dir.glob("*.parquet"))
        if prefix is None:
            return names
        return [name for name in names if name.startswith(prefix)]

    def read_layer_prefix(self, layer: str, prefix: str) -> pd.DataFrame:
        frames = [self.read_parquet(layer, name) for name in self.list_dataset_names(layer, prefix)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def query_parquet(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(sql, params or {}).fetchdf()

    def register_parquet(self, table_name: str, layer: str, name: str) -> None:
        if not table_name.replace("_", "").isalnum():
            raise ValueError("table_name must be alphanumeric or underscore")
        path = self.dataset_path(layer, name)
        escaped_path = str(path).replace("'", "''")
        sql = (
            f"CREATE OR REPLACE VIEW {table_name} "
            f"AS SELECT * FROM read_parquet('{escaped_path}')"
        )
        with self.connect() as con:
            con.execute(sql)
