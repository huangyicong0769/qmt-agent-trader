"""Stable, audited, pending-only storage migration registry."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb

from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import StorageConflictError, StorageMigrationFailedError


@dataclass(frozen=True)
class Migration:
    migration_id: str
    component: str
    version: int
    description: str
    apply: Callable[[duckdb.DuckDBPyConnection], Any]
    destructive: bool = False

    @property
    def checksum(self) -> str:
        payload = f"{self.migration_id}\0{self.component}\0{self.version}\0{self.description}"
        return hashlib.sha256(payload.encode()).hexdigest()


class MigrationRegistry:
    def __init__(self, coordinator: DatabaseCoordinator) -> None:
        self.coordinator = coordinator

    def apply(
        self, migrations: Iterable[Migration], *, dry_run: bool = False,
        allow_destructive: bool = False,
    ) -> list[str]:
        ordered = sorted(migrations, key=lambda item: (item.component, item.version))
        self._ensure_table()
        existing = self._existing()
        pending: list[Migration] = []
        for migration in ordered:
            prior = existing.get(migration.migration_id)
            if prior is not None:
                if prior[0] != migration.checksum:
                    raise StorageConflictError(
                        store_name="migrations", database_path=self.coordinator.database_path,
                        operation="verify_checksum", reason="migration checksum mismatch",
                    )
                if prior[1] == "APPLIED":
                    continue
            if migration.destructive and not allow_destructive:
                raise StorageConflictError(
                    store_name="migrations", database_path=self.coordinator.database_path,
                    operation="apply", reason="destructive migration requires explicit approval",
                )
            pending.append(migration)
        if dry_run:
            return [item.migration_id for item in pending]
        applied: list[str] = []
        for migration in pending:
            try:
                with self.coordinator.write_transaction(
                    f"migration:{migration.migration_id}"
                ) as connection:
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
            except Exception as exc:
                self._record_failure(migration, exc)
                raise StorageMigrationFailedError(
                    store_name="migrations", database_path=self.coordinator.database_path,
                    operation="apply", reason="migration failed", original_error=exc,
                ) from exc
            applied.append(migration.migration_id)
        return applied

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

    def _existing(self) -> dict[str, tuple[str, str]]:
        with self.coordinator.read_connection("read_migrations") as connection:
            rows = connection.execute(
                "SELECT migration_id, checksum, status FROM storage_schema_migrations"
            ).fetchall()
        return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}

    def _record_failure(self, migration: Migration, error: Exception) -> None:
        with self.coordinator.write_transaction("audit_failed_migration") as connection:
            connection.execute(
                "INSERT OR REPLACE INTO storage_schema_migrations VALUES "
                "(?, ?, ?, ?, 'FAILED', ?, ?, ?)",
                [migration.migration_id, migration.component, migration.version,
                 migration.checksum, _now(), _now(), type(error).__name__],
            )


def _now() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)
