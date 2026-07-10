"""Canonical inter-process lock management and ordering enforcement."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from filelock import FileLock, Timeout

from qmt_agent_trader.persistence.errors import StorageConflictError, StorageLockTimeoutError

_active_lock_kinds: ContextVar[tuple[str, ...]] = ContextVar("active_lock_kinds", default=())


class LockManager:
    def __init__(
        self,
        locks_root: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.locks_root = locks_root.expanduser().resolve()
        self.timeout_seconds = timeout_seconds

    @property
    def active_lock_kinds(self) -> tuple[str, ...]:
        return _active_lock_kinds.get()

    def lock_path_for_resource(self, resource_id: str | Path) -> Path:
        canonical = _canonical_resource(resource_id)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return self.locks_root / f"resource-{digest}.lock"

    @contextmanager
    def resource_lock(self, resource_id: str | Path) -> Iterator[FileLock]:
        if "database" in self.active_lock_kinds:
            raise StorageConflictError(
                store_name="locks",
                path=self.lock_path_for_resource(resource_id),
                operation="acquire_resource_lock",
                reason="lock order inversion: resource lock requested after database write lock",
            )
        with self._lock(self.lock_path_for_resource(resource_id), "resource") as lock:
            yield lock

    @contextmanager
    def database_write_lock(self, database_path: Path) -> Iterator[FileLock]:
        canonical = str(database_path.expanduser().resolve())
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        with self._lock(self.locks_root / f"database-{digest}.lock", "database") as lock:
            yield lock

    @contextmanager
    def _lock(self, path: Path, kind: str) -> Iterator[FileLock]:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path), timeout=self.timeout_seconds)
        try:
            lock.acquire()
        except Timeout as exc:
            operation = f"acquire_{kind}_lock"
            raise StorageLockTimeoutError(
                store_name="locks",
                path=path,
                operation=operation,
                reason=f"timed out after {self.timeout_seconds}s",
                recoverable=True,
                suggested_repair="retry after the current writer finishes",
                original_error=exc,
            ) from exc
        token = _active_lock_kinds.set((*self.active_lock_kinds, kind))
        try:
            yield lock
        finally:
            _active_lock_kinds.reset(token)
            lock.release()


def _canonical_resource(resource_id: str | Path) -> str:
    return str(Path(resource_id).expanduser().resolve())
