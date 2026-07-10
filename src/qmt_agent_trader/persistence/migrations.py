"""Stable, audited, pending-only storage migration registry."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb

from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageError,
    StorageMigrationFailedError,
)


@dataclass(frozen=True)
class Migration:
    migration_id: str
    component: str
    version: int
    description: str
    apply: Callable[[duckdb.DuckDBPyConnection], Any]
    destructive: bool = False
    implementation: str | bytes | None = None

    @property
    def checksum(self) -> str:
        metadata = (
            f"{self.migration_id}\0{self.component}\0{self.version}\0{self.description}"
        ).encode()
        implementation = self.implementation
        if implementation is None:
            code = self.apply.__code__
            implementation_bytes = code.co_code + repr(
                (code.co_consts, code.co_names)
            ).encode()
        elif isinstance(implementation, str):
            implementation_bytes = implementation.encode()
        else:
            implementation_bytes = implementation
        return hashlib.sha256(metadata + b"\0" + implementation_bytes).hexdigest()


class MigrationRegistry:
    def __init__(self, coordinator: DatabaseCoordinator) -> None:
        self.coordinator = coordinator

    def apply(
        self, migrations: Iterable[Migration], *, dry_run: bool = False,
        allow_destructive: bool = False,
    ) -> list[str]:
        ordered = sorted(migrations, key=lambda item: (item.component, item.version))
        if dry_run:
            existing = self._existing_for_dry_run()
            return [
                migration.migration_id
                for migration in ordered
                if self._is_pending(
                    migration, existing.get(migration.migration_id), allow_destructive
                )
            ]
        self._ensure_table()
        applied: list[str] = []
        for migration in ordered:
            try:
                with self.coordinator.write_transaction(
                    f"migration:{migration.migration_id}"
                ) as connection:
                    prior = connection.execute(
                        "SELECT checksum, status FROM storage_schema_migrations "
                        "WHERE migration_id = ?",
                        [migration.migration_id],
                    ).fetchone()
                    normalized_prior = (
                        (str(prior[0]), str(prior[1])) if prior is not None else None
                    )
                    if not self._is_pending(
                        migration, normalized_prior, allow_destructive
                    ):
                        continue
                    now = _now()
                    connection.execute(
                        "INSERT OR REPLACE INTO storage_schema_migrations VALUES "
                        "(?, ?, ?, ?, 'STARTED', ?, NULL, NULL)",
                        [migration.migration_id, migration.component, migration.version,
                         migration.checksum, now],
                    )
                    migration.apply(connection)
                    connection.execute(
                        "UPDATE storage_schema_migrations SET status='APPLIED', finished_at=? "
                        "WHERE migration_id=?", [_now(), migration.migration_id],
                    )
            except StorageConflictError:
                raise
            except Exception as exc:
                self._record_failure(migration, exc)
                raise StorageMigrationFailedError(
                    store_name="migrations", database_path=self.coordinator.database_path,
                    operation="apply", reason="migration failed", original_error=exc,
                ) from exc
            applied.append(migration.migration_id)
        return applied

    def _is_pending(
        self,
        migration: Migration,
        prior: tuple[str, str] | None,
        allow_destructive: bool,
    ) -> bool:
        if prior is not None:
            if prior[0] != migration.checksum:
                raise StorageConflictError(
                    store_name="migrations", database_path=self.coordinator.database_path,
                    operation="verify_checksum", reason="migration checksum mismatch",
                )
            if prior[1] == "APPLIED":
                return False
        if migration.destructive and not allow_destructive:
            raise StorageConflictError(
                store_name="migrations", database_path=self.coordinator.database_path,
                operation="apply", reason="destructive migration requires explicit approval",
            )
        return True

    def _ensure_table(self) -> None:
        with self.coordinator.write_transaction("initialize_migrations") as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS storage_schema_migrations (
                    migration_id TEXT PRIMARY KEY, component TEXT NOT NULL,
                    version INTEGER NOT NULL, checksum TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP,
                    error_message TEXT
                )"""
            )

    def _existing_for_dry_run(self) -> dict[str, tuple[str, str]]:
        if not self.coordinator.database_path.exists():
            return {}
        try:
            with self.coordinator.read_connection(
                "dry_run_migrations", read_only=True
            ) as connection:
                rows = connection.execute(
                    "SELECT migration_id, checksum, status FROM storage_schema_migrations"
                ).fetchall()
        except StorageError as exc:
            if isinstance(exc.__cause__, duckdb.CatalogException):
                return {}
            raise
        return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}

    def _record_failure(self, migration: Migration, error: Exception) -> None:
        with self.coordinator.write_transaction("audit_failed_migration") as connection:
            status = connection.execute(
                "SELECT status FROM storage_schema_migrations WHERE migration_id = ?",
                [migration.migration_id],
            ).fetchone()
            if status is not None and str(status[0]) == "APPLIED":
                return
            connection.execute(
                "INSERT OR REPLACE INTO storage_schema_migrations VALUES "
                "(?, ?, ?, ?, 'FAILED', ?, ?, ?)",
                [migration.migration_id, migration.component, migration.version,
                 migration.checksum, _now(), _now(), type(error).__name__],
            )


def _now() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)
