"""Short-lived DuckDB connections coordinated by one global write lock."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, Lock, get_ident
from time import monotonic, sleep
from typing import Any

import duckdb

from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageError,
    StorageLockTimeoutError,
)
from qmt_agent_trader.persistence.locks import LockManager

_gates_lock = Lock()
_database_gates: dict[Path, _ReadWriteGate] = {}


class DatabaseCoordinator:
    def __init__(
        self, database_path: Path, lock_manager: LockManager, *, store_name: str = "control_db"
    ) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.lock_manager = lock_manager
        self.store_name = store_name
        self._gate = _gate_for(self.database_path)

    @contextmanager
    def read_connection(
        self, operation: str = "read", *, read_only: bool | None = None
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        connection: duckdb.DuckDBPyConnection | None = None
        with self._shared_access(operation):
            try:
                effective_read_only = (
                    self.database_path.exists() if read_only is None else read_only
                )
                connection = self._open_connection(operation, read_only=effective_read_only)
                yield connection
            except StorageError:
                raise
            except duckdb.Error as exc:
                raise self._error(operation, exc) from exc
            finally:
                if connection is not None:
                    connection.close()

    @contextmanager
    def write_transaction(self, operation: str = "write") -> Iterator[duckdb.DuckDBPyConnection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_access(operation):
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
        with self._exclusive_access(operation):
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
        return StorageError(
            store_name=self.store_name,
            database_path=self.database_path,
            operation=operation,
            reason="database operation failed",
            original_error=error,
        )

    @contextmanager
    def _shared_access(self, operation: str) -> Iterator[None]:
        try:
            self._gate.acquire_read(self.lock_manager.timeout_seconds)
        except _GateTimeout as exc:
            raise self._gate_timeout(operation, exc) from exc
        except _GateNestedConflict as exc:
            raise self._gate_conflict(operation, exc) from exc
        try:
            yield
        finally:
            self._gate.release_read()

    @contextmanager
    def _exclusive_access(self, operation: str) -> Iterator[None]:
        try:
            self._gate.acquire_write(self.lock_manager.timeout_seconds)
        except _GateTimeout as exc:
            raise self._gate_timeout(operation, exc) from exc
        except _GateNestedConflict as exc:
            raise self._gate_conflict(operation, exc) from exc
        try:
            yield
        finally:
            self._gate.release_write()

    def _gate_timeout(self, operation: str, error: BaseException) -> StorageLockTimeoutError:
        return StorageLockTimeoutError(
            store_name=self.store_name,
            database_path=self.database_path,
            operation=operation,
            reason="timed out waiting for in-process database readers or writer",
            recoverable=True,
            suggested_repair="retry after the active local database operation finishes",
            original_error=error,
        )

    def _gate_conflict(self, operation: str, error: BaseException) -> StorageConflictError:
        return StorageConflictError(
            store_name=self.store_name,
            database_path=self.database_path,
            operation=operation,
            reason="nested database access would deadlock or mix connection configurations",
            recoverable=False,
            suggested_repair="reuse the active transaction connection",
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
    return "lock" in message and ("conflict" in message or "could not set lock" in message)


class _GateTimeout(RuntimeError):
    pass


class _GateNestedConflict(RuntimeError):
    pass


class _ReadWriteGate:
    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._reader_count = 0
        self._readers_by_thread: dict[int, int] = {}
        self._writer_thread: int | None = None
        self._waiting_writers = 0

    def acquire_read(self, timeout_seconds: float) -> None:
        thread_id = get_ident()
        deadline = monotonic() + timeout_seconds
        with self._condition:
            if self._writer_thread == thread_id:
                raise _GateNestedConflict("read requested inside active write transaction")
            already_reading = self._readers_by_thread.get(thread_id, 0) > 0
            while self._writer_thread is not None or (
                self._waiting_writers > 0 and not already_reading
            ):
                remaining = deadline - monotonic()
                if remaining <= 0 or not self._condition.wait(timeout=remaining):
                    raise _GateTimeout("read permit timed out")
            self._reader_count += 1
            self._readers_by_thread[thread_id] = self._readers_by_thread.get(thread_id, 0) + 1

    def release_read(self) -> None:
        thread_id = get_ident()
        with self._condition:
            count = self._readers_by_thread.get(thread_id, 0)
            if count <= 0:
                raise RuntimeError("read permit released by non-owner")
            if count == 1:
                del self._readers_by_thread[thread_id]
            else:
                self._readers_by_thread[thread_id] = count - 1
            self._reader_count -= 1
            if self._reader_count == 0:
                self._condition.notify_all()

    def acquire_write(self, timeout_seconds: float) -> None:
        thread_id = get_ident()
        deadline = monotonic() + timeout_seconds
        with self._condition:
            if self._writer_thread == thread_id or self._readers_by_thread.get(thread_id, 0):
                raise _GateNestedConflict("nested write requested by active database owner")
            self._waiting_writers += 1
            try:
                while self._writer_thread is not None or self._reader_count > 0:
                    remaining = deadline - monotonic()
                    if remaining <= 0 or not self._condition.wait(timeout=remaining):
                        raise _GateTimeout("write permit timed out")
                self._writer_thread = thread_id
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        thread_id = get_ident()
        with self._condition:
            if self._writer_thread != thread_id:
                raise RuntimeError("write permit released by non-owner")
            self._writer_thread = None
            self._condition.notify_all()


def _gate_for(database_path: Path) -> _ReadWriteGate:
    canonical = database_path.expanduser().resolve()
    with _gates_lock:
        gate = _database_gates.get(canonical)
        if gate is None:
            gate = _ReadWriteGate()
            _database_gates[canonical] = gate
        return gate
