"""DuckDB + Parquet data lake helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from qmt_agent_trader.data.catalog import is_legacy_raw_batch_name


@dataclass(frozen=True)
class LegacyMigrationResult:
    stable_name: str
    legacy_names: list[str]
    removed_names: list[str]
    rows: int

    def as_dict(self) -> dict[str, object]:
        return {
            "stable_name": self.stable_name,
            "legacy_names": self.legacy_names,
            "removed_names": self.removed_names,
            "rows": self.rows,
        }


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

    def write_incremental_parquet(
        self,
        frame: pd.DataFrame,
        layer: str,
        name: str,
        *,
        key_columns: list[str],
    ) -> Path:
        if frame.empty and len(frame.columns) == 0:
            frame = pd.DataFrame({column: pd.Series(dtype="object") for column in key_columns})
        missing = [column for column in key_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"incremental dataset missing key columns: {missing}")

        frames: list[pd.DataFrame] = []
        path = self.dataset_path(layer, name)
        if path.exists():
            existing = self.read_parquet(layer, name)
            if "_empty" in existing.columns:
                existing = existing.drop(columns=["_empty"])
            if not existing.empty:
                frames.append(existing)
        if not frame.empty:
            frames.append(frame.copy())

        if frames:
            merged = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(key_columns, keep="last")
                .sort_values(key_columns)
                .reset_index(drop=True)
            )
        else:
            merged = frame.copy()
        path = self.write_parquet(merged, layer, name)
        self.register_parquet(name, layer, name)
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

    def migrate_legacy_dataset(
        self,
        *,
        layer: str,
        stable_name: str,
        legacy_prefix: str,
        key_columns: list[str],
        remove_legacy: bool = True,
    ) -> LegacyMigrationResult:
        legacy_names = [
            name
            for name in self.list_dataset_names(layer, prefix=legacy_prefix)
            if name != stable_name and is_legacy_raw_batch_name(name)
        ]
        frames: list[pd.DataFrame] = []
        for name in legacy_names:
            frame = self.read_parquet(layer, name)
            if "_empty" in frame.columns:
                frame = frame.drop(columns=["_empty"])
            if not frame.empty:
                frames.append(frame)

        if self.dataset_path(layer, stable_name).exists():
            stable = self.read_parquet(layer, stable_name)
            if "_empty" in stable.columns:
                stable = stable.drop(columns=["_empty"])
            if not stable.empty:
                frames.append(stable)

        if frames:
            merged = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(key_columns, keep="last")
                .sort_values(key_columns)
                .reset_index(drop=True)
            )
            self.write_incremental_parquet(
                merged,
                layer,
                stable_name,
                key_columns=key_columns,
            )
        elif legacy_names:
            self.write_incremental_parquet(
                pd.DataFrame(),
                layer,
                stable_name,
                key_columns=key_columns,
            )

        removed_names: list[str] = []
        if remove_legacy:
            for name in legacy_names:
                path = self.dataset_path(layer, name)
                if path.exists():
                    path.unlink()
                    removed_names.append(name)

        rows = 0
        if self.dataset_path(layer, stable_name).exists():
            rows = len(self.read_parquet(layer, stable_name))
        return LegacyMigrationResult(
            stable_name=stable_name,
            legacy_names=legacy_names,
            removed_names=removed_names,
            rows=rows,
        )

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

    def ensure_fetch_tables(self) -> None:
        with self.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS data_fetch_state (
                    source TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    row_count BIGINT NOT NULL,
                    checksum TEXT,
                    updated_at TIMESTAMP NOT NULL,
                    error TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS data_fetch_events (
                    source TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    row_count BIGINT NOT NULL,
                    checksum TEXT,
                    updated_at TIMESTAMP NOT NULL,
                    error TEXT
                )
                """
            )

    def record_fetch_result(
        self,
        *,
        source: str,
        dataset: str,
        start_date: str,
        end_date: str,
        status: str,
        row_count: int,
        checksum: str | None,
        error: str | None,
    ) -> None:
        self.ensure_fetch_tables()
        updated_at = datetime.now(tz=UTC).replace(tzinfo=None)
        values = [
            source,
            dataset,
            start_date,
            end_date,
            status,
            row_count,
            checksum,
            updated_at,
            error,
        ]
        with self.connect() as con:
            con.execute(
                """
                DELETE FROM data_fetch_state
                WHERE source = ?
                  AND dataset = ?
                  AND start_date = ?
                  AND end_date = ?
                """,
                [source, dataset, start_date, end_date],
            )
            con.execute(
                """
                INSERT INTO data_fetch_state
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                values,
            )
            con.execute(
                """
                INSERT INTO data_fetch_events
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                values,
            )

    def fetch_state(self, source: str, dataset: str) -> list[dict[str, Any]]:
        self.ensure_fetch_tables()
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT source, dataset, start_date, end_date, status, row_count, checksum, error
                FROM data_fetch_state
                WHERE source = ? AND dataset = ?
                ORDER BY start_date, end_date
                """,
                [source, dataset],
            ).fetchdf()
        return cast(list[dict[str, Any]], rows.to_dict(orient="records"))

    def fetch_events(self, source: str, dataset: str) -> list[dict[str, Any]]:
        self.ensure_fetch_tables()
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT source, dataset, start_date, end_date, status, row_count, checksum, error
                FROM data_fetch_events
                WHERE source = ? AND dataset = ?
                ORDER BY updated_at
                """,
                [source, dataset],
            ).fetchdf()
        return cast(list[dict[str, Any]], rows.to_dict(orient="records"))
