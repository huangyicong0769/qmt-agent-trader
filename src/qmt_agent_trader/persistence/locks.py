"""Canonical inter-process lock management and ordering enforcement."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

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
        with self._writer_admission("resource", resource_id):
            with self._lock(self.lock_path_for_resource(resource_id), "resource") as lock:
                yield lock

    @contextmanager
    def database_write_lock(self, database_path: Path) -> Iterator[FileLock]:
        canonical = str(database_path.expanduser().resolve())
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        with self._writer_admission("database", database_path):
            with self._lock(self.locks_root / f"database-{digest}.lock", "database") as lock:
                yield lock

    @contextmanager
    def backup_barrier(self) -> Iterator[FileLock]:
        """Exclude all cooperating filesystem and database writers."""
        if any(kind in self.active_lock_kinds for kind in ("backup", "resource", "database")):
            raise StorageConflictError(
                store_name="locks",
                path=self.locks_root / "maintenance.lock",
                operation="acquire_backup_barrier",
                reason="backup cannot start from an active writer or backup context",
            )
        maintenance_path = self.locks_root / "maintenance.active"
        with self._lock(self.locks_root / "writer-admission.lock", "admission"):
            self._create_marker(
                maintenance_path,
                operation="backup",
                resource="all-authoritative-stores",
            )
        token = _active_lock_kinds.set((*self.active_lock_kinds, "backup"))
        try:
            self._wait_for_active_writers()
            yield FileLock(str(self.locks_root / "maintenance.lock"))
        finally:
            _active_lock_kinds.reset(token)
            maintenance_path.unlink(missing_ok=True)

    @contextmanager
    def _writer_admission(
        self, operation: str, resource: str | Path
    ) -> Iterator[Path | None]:
        if "backup" in self.active_lock_kinds:
            yield None
            return
        deadline = monotonic() + self.timeout_seconds
        marker: Path | None = None
        while marker is None:
            with self._lock(
                self.locks_root / "writer-admission.lock",
                "admission",
                error_kind=operation,
            ):
                if not (self.locks_root / "maintenance.active").exists():
                    marker = self._new_writer_marker(operation, resource)
            if marker is None:
                if monotonic() >= deadline:
                    raise StorageLockTimeoutError(
                        store_name="locks",
                        path=self.locks_root / "maintenance.active",
                        operation=f"acquire_{operation}_lock",
                        reason="timed out waiting for storage maintenance to finish",
                        recoverable=True,
                        suggested_repair="retry after backup or maintenance finishes",
                    )
                sleep(min(0.05, max(deadline - monotonic(), 0.0)))
        try:
            yield marker
        finally:
            marker.unlink(missing_ok=True)

    def _new_writer_marker(self, operation: str, resource: str | Path) -> Path:
        marker = self.locks_root / "writers" / f"{os.getpid()}-{uuid4().hex}.json"
        self._create_marker(marker, operation=operation, resource=str(resource))
        return marker

    def _create_marker(self, path: Path, *, operation: str, resource: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "operation": operation,
                "started_at": datetime.now(tz=UTC).isoformat(),
                "resource": resource,
            },
            sort_keys=True,
        ).encode()
        with path.open("xb") as handle:
            handle.write(payload)

    def _wait_for_active_writers(self) -> None:
        deadline = monotonic() + self.timeout_seconds
        writers_root = self.locks_root / "writers"
        while True:
            active = [path for path in writers_root.glob("*.json") if self._marker_is_active(path)]
            if not active:
                return
            if monotonic() >= deadline:
                raise StorageLockTimeoutError(
                    store_name="locks",
                    path=writers_root,
                    operation="acquire_backup_lock",
                    reason=f"timed out waiting for {len(active)} active writer(s)",
                    recoverable=True,
                    suggested_repair="retry after active writers finish",
                )
            sleep(min(0.05, max(deadline - monotonic(), 0.0)))

    def _marker_is_active(self, path: Path) -> bool:
        try:
            payload = json.loads(path.read_text())
            pid = int(payload["pid"])
            host = str(payload["host"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return True
        if host != socket.gethostname():
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            return False
        except PermissionError:
            return True
        return True

    @contextmanager
    def _lock(self, path: Path, kind: str, *, error_kind: str | None = None) -> Iterator[FileLock]:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path), timeout=self.timeout_seconds)
        try:
            lock.acquire()
        except Timeout as exc:
            operation = f"acquire_{error_kind or kind}_lock"
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
