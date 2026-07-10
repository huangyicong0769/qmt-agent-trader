"""DuckDB + Parquet data lake helpers."""

from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from qmt_agent_trader.data.atomic_io import atomic_write_parquet
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import (
    StorageLockTimeoutError,
    StorageMigrationRequiredError,
)
from qmt_agent_trader.persistence.locks import LockManager


class DataLakeLockTimeoutError(RuntimeError):
    def __init__(self, path: Path, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Timed out after {timeout_seconds}s waiting for Parquet dataset lock: {path}"
        )


class DataLake:
    def __init__(
        self,
        root: Path,
        duckdb_path: Path,
        *,
        parquet_lock_timeout_seconds: float = 30.0,
        database_coordinator: DatabaseCoordinator | None = None,
        lock_manager: LockManager | None = None,
    ) -> None:
        resolved_database_path = duckdb_path.expanduser().resolve()
        if (
            database_coordinator is not None
            and database_coordinator.database_path != resolved_database_path
        ):
            raise ValueError("injected coordinator database path does not match duckdb_path")
        if (
            database_coordinator is not None
            and lock_manager is not None
            and database_coordinator.lock_manager is not lock_manager
        ):
            raise ValueError("injected coordinator and lock manager must match")
        self.root = root
        self.duckdb_path = duckdb_path
        self.parquet_lock_timeout_seconds = parquet_lock_timeout_seconds
        self.lock_manager = lock_manager or (
            database_coordinator.lock_manager
            if database_coordinator is not None
            else LockManager(
                self.duckdb_path.parent / "locks",
                timeout_seconds=parquet_lock_timeout_seconds,
            )
        )
        self.database_coordinator = database_coordinator or DatabaseCoordinator(
            self.duckdb_path, self.lock_manager
        )
        self._persistence_schema_initialized = False
        self._persistence_schema_attempted = False
        self._legacy_ledger_initialized = False
        self._persistence_schema_error: Exception | None = None
        self._legacy_ledger_error: Exception | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> AbstractContextManager[duckdb.DuckDBPyConnection]:
        """Compatibility read connection; writes use the injected coordinator."""
        return self.database_coordinator.read_connection("data_lake_read")

    @property
    def persistence_initialized(self) -> bool:
        return (
            self._persistence_schema_initialized
            and self._legacy_ledger_initialized
            and self.persistence_initialization_error is None
        )

    @property
    def persistence_schema_initialized(self) -> bool:
        return self._persistence_schema_initialized

    @property
    def legacy_ledger_initialized(self) -> bool:
        return self._legacy_ledger_initialized

    @property
    def persistence_initialization_error(self) -> Exception | None:
        return self._persistence_schema_error or self._legacy_ledger_error

    @property
    def persistence_schema_attempted(self) -> bool:
        return self._persistence_schema_attempted

    @property
    def persistence_schema_error(self) -> Exception | None:
        return self._persistence_schema_error

    @property
    def legacy_ledger_error(self) -> Exception | None:
        return self._legacy_ledger_error

    def mark_persistence_schema_initialized(self) -> None:
        self._persistence_schema_attempted = True
        self._persistence_schema_initialized = True
        self._persistence_schema_error = None

    def mark_persistence_schema_failed(self, error: Exception) -> None:
        self._persistence_schema_attempted = True
        self._persistence_schema_initialized = False
        self._persistence_schema_error = error

    def mark_legacy_ledger_initialized(self, *, error: Exception | None = None) -> None:
        self._legacy_ledger_initialized = True
        self._legacy_ledger_error = error

    def dataset_path(self, layer: str, name: str) -> Path:
        return self.root / layer / f"{name}.parquet"

    def dataset_path_for_id(self, layer: str, dataset_id: str) -> Path:
        return self.dataset_path(layer, dataset_name_from_id(dataset_id))

    def register_dataset_id(self, dataset_id: str, layer: str, name: str) -> None:
        self.register_parquet(dataset_view_name(layer, dataset_id), layer, name)

    def write_parquet(self, frame: pd.DataFrame, layer: str, name: str) -> Path:
        path = self.dataset_path(layer, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = frame
        if len(frame.columns) == 0:
            writable = pd.DataFrame({"_empty": pd.Series(dtype="bool")})
        atomic_write_parquet(writable, path)
        return path

    def write_incremental_parquet(
        self,
        frame: pd.DataFrame,
        layer: str,
        name: str,
        *,
        key_columns: list[str],
    ) -> Path:
        path = self.dataset_path(layer, name)
        try:
            with self.lock_manager.resource_lock(path):
                if frame.empty and len(frame.columns) == 0:
                    frame = pd.DataFrame(
                        {column: pd.Series(dtype="object") for column in key_columns}
                    )
                missing = [column for column in key_columns if column not in frame.columns]
                if missing:
                    raise ValueError(f"incremental dataset missing key columns: {missing}")

                frames: list[pd.DataFrame] = []
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
                path.parent.mkdir(parents=True, exist_ok=True)
                writable = merged
                if len(merged.columns) == 0:
                    writable = pd.DataFrame({"_empty": pd.Series(dtype="bool")})
                atomic_write_parquet(writable, path)
                table_name = name if name.replace("_", "").isalnum() else name.replace("/", "_")
                self.register_parquet(table_name, layer, name)
        except StorageLockTimeoutError as exc:
            raise DataLakeLockTimeoutError(path, self.parquet_lock_timeout_seconds) from exc
        return path

    def write_incremental_dataset(
        self,
        frame: pd.DataFrame,
        *,
        layer: str,
        dataset_id: str,
        name: str,
        key_columns: list[str],
    ) -> Path:
        path = self.write_incremental_parquet(
            frame,
            layer,
            name,
            key_columns=key_columns,
        )
        self.register_dataset_id(dataset_id, layer, name)
        return path

    def read_parquet(self, layer: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.dataset_path(layer, name))

    def read_parquet_filtered(
        self,
        layer: str,
        name: str,
        *,
        columns: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        date_column: str = "trade_date",
        symbols: list[str] | None = None,
        symbol_column: str = "ts_code",
    ) -> pd.DataFrame:
        path = self.dataset_path(layer, name)
        if not path.exists():
            return pd.DataFrame()

        escaped_path = str(path).replace("'", "''")
        with self.connect() as con:
            schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')").fetchdf()
            available_columns = [str(item) for item in schema["column_name"].tolist()]
            available = set(available_columns)
            selected_columns = (
                [column for column in columns if column in available]
                if columns is not None
                else available_columns
            )
            if columns is not None and not selected_columns:
                return pd.DataFrame(columns=columns)

            predicates: list[str] = []
            params: list[object] = []
            if date_column in available and start is not None:
                predicates.append(f"{_date_key_sql(date_column)} >= ?")
                params.append(_date_key(start))
            if date_column in available and end is not None:
                predicates.append(f"{_date_key_sql(date_column)} <= ?")
                params.append(_date_key(end))
            normalized_symbols = _normalized_symbols(symbols)
            if symbol_column in available and normalized_symbols:
                placeholders = ", ".join("?" for _ in normalized_symbols)
                predicates.append(f"{_quote_identifier(symbol_column)} IN ({placeholders})")
                params.extend(normalized_symbols)

            select_sql = ", ".join(_quote_identifier(column) for column in selected_columns)
            sql = f"SELECT {select_sql} FROM read_parquet('{escaped_path}')"
            if predicates:
                sql += " WHERE " + " AND ".join(predicates)
            return con.execute(sql, params).fetchdf()

    def list_dataset_names(self, layer: str, prefix: str | None = None) -> list[str]:
        layer_dir = self.root / layer
        if not layer_dir.exists():
            return []
        names = sorted(
            path.relative_to(layer_dir).with_suffix("").as_posix()
            for path in layer_dir.rglob("*.parquet")
        )
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
        sql = f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{escaped_path}')"
        with self.database_coordinator.write_transaction("register_parquet") as con:
            con.execute(sql)

    def ensure_fetch_tables(self) -> None:
        """Compatibility initialization entry point; normal reads never call it."""
        from qmt_agent_trader.persistence.initialization import initialize_persistence

        initialize_persistence(self, migrate_legacy_ledger=False)

    def record_fetch_metadata(
        self,
        *,
        source: str,
        dataset_id: str,
        api_name: str,
        endpoint_id: str,
        params: dict[str, Any],
        fields: list[str],
        symbols: list[str],
        coverage_start: str | None,
        coverage_end: str | None,
        row_count: int,
        checksum: str | None,
        status: str,
        error: str | None,
    ) -> None:
        self._assert_fetch_tables_ready()
        fetched_at = datetime.now(tz=UTC).replace(tzinfo=None)
        values = [
            source,
            dataset_id,
            api_name,
            endpoint_id,
            _stable_hash(params),
            _stable_hash(fields),
            _stable_hash(symbols),
            fetched_at,
            coverage_start,
            coverage_end,
            row_count,
            checksum,
            status,
            error,
        ]
        with self.database_coordinator.write_transaction("record_fetch_metadata") as con:
            con.execute(
                """
                INSERT INTO data_fetch_state_v2
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    source, dataset_id, api_name, endpoint_id,
                    param_hash, fields_hash, symbols_hash
                ) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    coverage_start = excluded.coverage_start,
                    coverage_end = excluded.coverage_end,
                    row_count = excluded.row_count,
                    checksum = excluded.checksum,
                    status = excluded.status,
                    error = excluded.error
                """,
                values,
            )
            con.execute(
                """
                INSERT INTO data_fetch_events_v2
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
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
        self._assert_fetch_tables_ready()
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
        with self.database_coordinator.write_transaction("record_fetch_result") as con:
            con.execute(
                """
                INSERT INTO data_fetch_state
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source, dataset, start_date, end_date) DO UPDATE SET
                    status = excluded.status,
                    row_count = excluded.row_count,
                    checksum = excluded.checksum,
                    updated_at = excluded.updated_at,
                    error = excluded.error
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
        self._assert_fetch_tables_ready()
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
        self._assert_fetch_tables_ready()
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

    def _assert_fetch_tables_ready(self) -> None:
        if self._persistence_schema_initialized:
            return
        required = {
            "data_fetch_state_v2",
            "data_fetch_events_v2",
            "data_fetch_state",
            "data_fetch_events",
        }
        if not self.duckdb_path.exists():
            raise self._fetch_schema_required()
        with self.database_coordinator.read_connection(
            "assert_fetch_schema", read_only=True
        ) as con:
            rows = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name IN (?, ?, ?, ?)",
                sorted(required),
            ).fetchall()
        if {str(row[0]) for row in rows} != required:
            raise self._fetch_schema_required()

    def _fetch_schema_required(self) -> StorageMigrationRequiredError:
        return StorageMigrationRequiredError(
            store_name="data_lake",
            database_path=self.duckdb_path,
            operation="assert_fetch_schema",
            reason="fetch metadata schema is not initialized",
            recoverable=True,
            suggested_repair="initialize persistence before reading or writing fetch metadata",
        )


def _quote_identifier(identifier: str) -> str:
    if "\x00" in identifier:
        raise ValueError("identifier contains null byte")
    return '"' + identifier.replace('"', '""') + '"'


def _date_key(value: str | date) -> str:
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value)
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return datetime.fromisoformat(text).strftime("%Y%m%d")


def _date_key_sql(column: str) -> str:
    column_sql = _quote_identifier(column)
    text = f"CAST({column_sql} AS VARCHAR)"
    return (
        "COALESCE("
        f"strftime(try_strptime({text}, '%Y%m%d'), '%Y%m%d'), "
        f"strftime(TRY_CAST({column_sql} AS DATE), '%Y%m%d'), "
        f"substr(regexp_replace({text}, '[^0-9]', '', 'g'), 1, 8)"
        ")"
    )


def _normalized_symbols(symbols: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw in symbols or []:
        symbol = str(raw).strip()
        if not symbol:
            continue
        if symbol not in normalized:
            normalized.append(symbol)
    return normalized


def dataset_name_from_id(dataset_id: str) -> str:
    parts = dataset_id.split(".")
    if len(parts) == 1:
        return dataset_id
    return "/".join(parts)


def dataset_view_name(layer: str, dataset_id: str) -> str:
    return f"{layer}_{dataset_id.replace('.', '_').replace('/', '_')}"


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
