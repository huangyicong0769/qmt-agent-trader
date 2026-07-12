"""Short-lived DuckDB connections coordinated by one global write lock."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import duckdb

from qmt_agent_trader.persistence.errors import (
    StorageCorruptError,
    StorageError,
    StorageLockTimeoutError,
    StoragePermissionError,
    StorageSchemaMismatchError,
    StorageUnavailableError,
)
from qmt_agent_trader.persistence.locks import LockManager


class DatabaseCoordinator:
    def __init__(
        self, database_path: Path, lock_manager: LockManager, *, store_name: str = "control_db"
    ) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.lock_manager = lock_manager
        self.store_name = store_name

    @contextmanager
    def read_connection(
        self, operation: str = "read", *, read_only: bool = True
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        if not read_only:
            raise StorageUnavailableError(
                store_name=self.store_name,
                database_path=self.database_path,
                operation=operation,
                reason="read connections must be read-only",
                suggested_repair="use write_transaction for mutations",
            )
        if not self.database_path.exists():
            raise StorageUnavailableError(
                store_name=self.store_name,
                database_path=self.database_path,
                operation=operation,
                reason="database is not initialized",
                suggested_repair="initialize persistence before reading",
            )
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = self._open_connection(operation, read_only=True)
            yield connection
        except StorageError:
            raise
        except duckdb.Error as exc:
            raise self._error(operation, exc) from exc
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def transient_read_connection(
        self, operation: str = "transient_read"
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        """Open an isolated in-memory query engine without touching control state."""
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = duckdb.connect(":memory:")
            yield connection
        except duckdb.Error as exc:
            raise self._error(operation, exc) from exc
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def write_transaction(self, operation: str = "write") -> Iterator[duckdb.DuckDBPyConnection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_manager.database_write_lock(self.database_path):
            connection: duckdb.DuckDBPyConnection | None = None
            try:
                connection = self._open_connection(operation, read_only=False)
                connection.execute("BEGIN TRANSACTION")
                yield connection
                connection.execute("COMMIT")
            except Exception as exc:
                if connection is not None:
                    try:
                        connection.execute("ROLLBACK")
                    except duckdb.Error:
                        pass
                if isinstance(exc, duckdb.Error):
                    raise self._error(operation, exc) from exc
                raise
            finally:
                if connection is not None:
                    connection.close()

    @contextmanager
    def write_connection(self, operation: str = "write") -> Iterator[duckdb.DuckDBPyConnection]:
        """Open a serialized autocommit connection for legacy statement semantics."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_manager.database_write_lock(self.database_path):
            connection: duckdb.DuckDBPyConnection | None = None
            try:
                connection = self._open_connection(operation, read_only=False)
                yield connection
            except duckdb.Error as exc:
                raise self._error(operation, exc) from exc
            finally:
                if connection is not None:
                    connection.close()

    def initialize(
        self, initializer: Callable[[duckdb.DuckDBPyConnection], Any] | None = None
    ) -> None:
        with self.write_transaction("initialize") as connection:
            if initializer is not None:
                initializer(connection)

    def checkpoint_copy(self, target: Path) -> None:
        """Checkpoint and copy a stable DuckDB file while writers are excluded."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.write_connection("checkpoint_backup") as connection:
            connection.execute("CHECKPOINT")
            shutil.copy2(self.database_path, target)

    def current_schema_version(self, component: str | None = None) -> int:
        try:
            with self.read_connection("current_schema_version") as connection:
                where = (
                    " WHERE component = ? AND status = 'APPLIED'"
                    if component
                    else (" WHERE status = 'APPLIED'")
                )
                params = [component] if component else []
                result = connection.execute(
                    "SELECT coalesce(max(version), 0) FROM storage_schema_migrations" + where,
                    params,
                ).fetchone()
        except StorageError as exc:
            cause = exc.__cause__
            if (
                isinstance(cause, duckdb.CatalogException)
                and "storage_schema_migrations" in str(cause)
                and "does not exist" in str(cause)
            ):
                return 0
            raise
        return int(result[0]) if result else 0

    def _error(self, operation: str, error: BaseException) -> StorageError:
        if (
            isinstance(error, duckdb.PermissionException)
            or "permission denied" in str(error).lower()
        ):
            return StoragePermissionError(
                store_name=self.store_name,
                database_path=self.database_path,
                operation=operation,
                reason="database permission denied",
                suggested_repair="correct database path permissions",
                original_error=error,
            )
        if isinstance(error, duckdb.CatalogException):
            return StorageSchemaMismatchError(
                store_name=self.store_name,
                database_path=self.database_path,
                operation=operation,
                reason="database schema object is missing or incompatible",
                suggested_repair="run persistence initialization or migrations",
                original_error=error,
            )
        if any(token in str(error).lower() for token in ("corrupt", "checksum mismatch")):
            return StorageCorruptError(
                store_name=self.store_name,
                database_path=self.database_path,
                operation=operation,
                reason="database integrity failure",
                suggested_repair="restore a verified control database backup",
                original_error=error,
            )
        return StorageError(
            store_name=self.store_name,
            database_path=self.database_path,
            operation=operation,
            reason="database operation failed",
            original_error=error,
        )

    def _open_connection(self, operation: str, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        deadline = monotonic() + self.lock_manager.timeout_seconds
        while True:
            try:
                return duckdb.connect(str(self.database_path), read_only=read_only)
            except duckdb.Error as exc:
                if not _is_database_lock_conflict(exc):
                    raise
                if monotonic() >= deadline:
                    raise StorageLockTimeoutError(
                        store_name=self.store_name,
                        database_path=self.database_path,
                        operation=operation,
                        reason=("timed out waiting for a compatible DuckDB process connection"),
                        recoverable=True,
                        suggested_repair="retry after active database readers or writers finish",
                        original_error=exc,
                    ) from exc
                sleep(min(0.05, max(deadline - monotonic(), 0.0)))


def _is_database_lock_conflict(error: duckdb.Error) -> bool:
    message = str(error).lower()
    return (
        "lock" in message and ("conflict" in message or "could not set lock" in message)
    ) or "different configuration" in message
