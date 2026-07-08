"""DuckDB + Parquet data lake helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

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
        table_name = name if name.replace("_", "").isalnum() else name.replace("/", "_")
        self.register_parquet(table_name, layer, name)
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
            schema = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')"
            ).fetchdf()
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
                CREATE TABLE IF NOT EXISTS data_fetch_state_v2 (
                    source TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    api_name TEXT NOT NULL,
                    endpoint_id TEXT NOT NULL,
                    param_hash TEXT NOT NULL,
                    fields_hash TEXT NOT NULL,
                    symbols_hash TEXT NOT NULL,
                    fetched_at TIMESTAMP NOT NULL,
                    coverage_start TEXT,
                    coverage_end TEXT,
                    row_count BIGINT NOT NULL,
                    checksum TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS data_fetch_events_v2 (
                    source TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    api_name TEXT NOT NULL,
                    endpoint_id TEXT NOT NULL,
                    param_hash TEXT NOT NULL,
                    fields_hash TEXT NOT NULL,
                    symbols_hash TEXT NOT NULL,
                    fetched_at TIMESTAMP NOT NULL,
                    coverage_start TEXT,
                    coverage_end TEXT,
                    row_count BIGINT NOT NULL,
                    checksum TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                )
                """
            )
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
        self.ensure_fetch_tables()
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
        with self.connect() as con:
            con.execute(
                """
                DELETE FROM data_fetch_state_v2
                WHERE source = ?
                  AND dataset_id = ?
                  AND api_name = ?
                  AND endpoint_id = ?
                  AND param_hash = ?
                  AND fields_hash = ?
                  AND symbols_hash = ?
                """,
                values[:7],
            )
            con.execute(
                """
                INSERT INTO data_fetch_state_v2
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
