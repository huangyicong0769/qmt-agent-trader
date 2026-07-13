"""Local storage inventory, health, migration, backup, and quarantine operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pyarrow.parquet as pq
import yaml

from qmt_agent_trader.persistence.artifacts import ArtifactStore, artifact_store_for_root
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.audit import AuditJsonlStore
from qmt_agent_trader.persistence.catalog import StoreCatalog, StoreDefinition, StoreLayout
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import (
    StorageBackupError,
    StorageConflictError,
    StorageCorruptError,
    StorageError,
    StorageLockTimeoutError,
    StoragePermissionError,
    StorageSchemaMismatchError,
    StorageUnavailableError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.initialization import storage_migrations
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.migrations import MigrationRegistry
from qmt_agent_trader.persistence.paths import PersistencePaths
from qmt_agent_trader.services.order_plan_service import (
    verify_bound_order_plan_event_stream_assume_locked,
)


@dataclass(frozen=True)
class StoreInventory:
    name: str
    type: str
    path: Path
    owner: str
    source_of_truth: str
    schema_version: int | None
    mutable: bool
    layout: StoreLayout
    lock_policy: str
    backup_policy: str
    health: str


@dataclass(frozen=True)
class StorageDiagnostic:
    component: str
    code: str
    reason: str
    path: Path | None = None
    severity: Literal["warning", "error"] = "error"


@dataclass(frozen=True)
class VerificationResult:
    healthy: bool
    deep: bool
    diagnostics: list[StorageDiagnostic]


@dataclass(frozen=True)
class BackupReceipt:
    path: Path
    manifest_path: Path


@dataclass(frozen=True)
class QuarantineReceipt:
    path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ResetPlan:
    status: Literal["planned"]
    profile: Literal["preserve-raw"]
    digest: str
    file_count: int
    byte_count: int
    preserved_raw_count: int
    preserved_raw_bytes: int
    preserved_raw_digest: str
    delete_paths: tuple[str, ...]


@dataclass(frozen=True)
class ResetReceipt:
    status: Literal["completed", "rolled_back", "rollback_failed"]
    profile: Literal["preserve-raw"]
    digest: str
    file_count: int
    byte_count: int
    preserved_raw_count: int
    preserved_raw_bytes: int
    preserved_raw_digest: str
    receipt_path: Path | None = None
    reason: str | None = None
    staging_path: Path | None = None


class StorageOperations:
    def __init__(self, paths: PersistencePaths, *, timeout_seconds: float = 30.0) -> None:
        self.paths = paths
        self.locks = LockManager(paths.locks_root, timeout_seconds=timeout_seconds)
        self.atomic = AtomicFileStore(self.locks)
        self.database = DatabaseCoordinator(paths.control_db_path, self.locks)
        self.catalog = StoreCatalog.canonical(paths)

    def inventory(self) -> list[StoreInventory]:
        return [
            StoreInventory(
                name=store.name,
                type=store.kind,
                path=store.path,
                owner=store.owner,
                source_of_truth=store.source_of_truth,
                schema_version=store.schema_version,
                mutable=store.mutable,
                layout=store.layout,
                lock_policy=store.lock_resource,
                backup_policy=store.backup,
                health="present" if store.path.exists() else "not_initialized",
            )
            for store in self.catalog.stores
        ]

    def plan_reset(self, *, profile: str) -> ResetPlan:
        if profile != "preserve-raw":
            raise StorageValidationError(
                store_name="storage_reset",
                path=self.paths.project_root,
                operation="plan_reset",
                reason=f"unsupported reset profile: {profile}",
            )
        raw_files = self._snapshot_reset_files(self.paths.lake_root / "raw", validate_parquet=True)
        delete_files: list[tuple[str, int, str]] = []
        delete_paths: list[str] = []
        for target in self._reset_targets():
            if not target.exists() and not target.is_symlink():
                continue
            self._validate_reset_target(target)
            delete_paths.append(target.relative_to(self.paths.project_root).as_posix())
            delete_files.extend(self._snapshot_reset_files(target))
        payload = {
            "schema_version": 1,
            "profile": profile,
            "delete_paths": sorted(delete_paths),
            "delete_files": sorted(delete_files),
            "preserved_raw": sorted(raw_files),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        raw_digest = hashlib.sha256(
            json.dumps(sorted(raw_files), separators=(",", ":")).encode()
        ).hexdigest()
        return ResetPlan(
            status="planned",
            profile="preserve-raw",
            digest=digest,
            file_count=len(delete_files),
            byte_count=sum(item[1] for item in delete_files),
            preserved_raw_count=len(raw_files),
            preserved_raw_bytes=sum(item[1] for item in raw_files),
            preserved_raw_digest=raw_digest,
            delete_paths=tuple(sorted(delete_paths)),
        )

    def reset(self, *, profile: str, confirm: str) -> ResetReceipt:
        staging = self.paths.project_root / f".storage-reset-{uuid4().hex}.staging"
        moved: list[tuple[Path, Path]] = []
        receipt_path: Path | None = None
        with self.locks.backup_barrier():
            plan = self.plan_reset(profile=profile)
            if not confirm or confirm != plan.digest:
                raise StorageValidationError(
                    store_name="storage_reset",
                    path=self.paths.project_root,
                    operation="reset",
                    reason="confirmation digest does not match the current reset plan",
                )
            raw_before = self._snapshot_reset_files(
                self.paths.lake_root / "raw", validate_parquet=True
            )
            try:
                staging.mkdir(parents=False, exist_ok=False)
                for relative in plan.delete_paths:
                    source = self.paths.project_root / relative
                    if not source.exists():
                        raise StorageConflictError(
                            store_name="storage_reset",
                            path=source,
                            operation="reset",
                            reason="reset target changed after confirmation",
                        )
                    destination = staging / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
                    moved.append((source, destination))
                MigrationRegistry(self.database).apply(storage_migrations())
                raw_after = self._snapshot_reset_files(
                    self.paths.lake_root / "raw", validate_parquet=True
                )
                if raw_after != raw_before:
                    raise StorageConflictError(
                        store_name="storage_reset",
                        path=self.paths.lake_root / "raw",
                        operation="reset",
                        reason="preserved raw snapshot changed during reset",
                    )
                verification = self.verify(deep=True)
                if not verification.healthy:
                    raise StorageValidationError(
                        store_name="storage_reset",
                        path=self.paths.project_root,
                        operation="reset",
                        reason="post-reset deep verification failed",
                    )
                receipt_path = (
                    self.paths.data_root
                    / "storage-resets"
                    / f"{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
                )
                self.atomic.write_json(
                    receipt_path,
                    {
                        "schema_version": 1,
                        "status": "completed",
                        "profile": plan.profile,
                        "digest": plan.digest,
                        "completed_at": datetime.now(tz=UTC).isoformat(),
                        "deleted_file_count": plan.file_count,
                        "deleted_byte_count": plan.byte_count,
                        "preserved_raw_count": plan.preserved_raw_count,
                        "preserved_raw_bytes": plan.preserved_raw_bytes,
                        "preserved_raw_digest": plan.preserved_raw_digest,
                        "verification": {
                            "healthy": verification.healthy,
                            "deep": verification.deep,
                            "diagnostics": len(verification.diagnostics),
                        },
                    },
                    create_only=True,
                )
                shutil.rmtree(staging)
                return ResetReceipt(
                    "completed",
                    plan.profile,
                    plan.digest,
                    plan.file_count,
                    plan.byte_count,
                    plan.preserved_raw_count,
                    plan.preserved_raw_bytes,
                    plan.preserved_raw_digest,
                    receipt_path,
                )
            except Exception as exc:
                try:
                    if receipt_path is not None:
                        receipt_path.unlink(missing_ok=True)
                    self._remove_new_reset_state()
                    for source, destination in reversed(moved):
                        source.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(destination, source)
                    shutil.rmtree(staging, ignore_errors=True)
                except Exception as rollback_exc:
                    return ResetReceipt(
                        "rollback_failed",
                        plan.profile,
                        plan.digest,
                        plan.file_count,
                        plan.byte_count,
                        plan.preserved_raw_count,
                        plan.preserved_raw_bytes,
                        plan.preserved_raw_digest,
                        reason=f"{type(exc).__name__}; rollback: {type(rollback_exc).__name__}",
                        staging_path=staging,
                    )
                return ResetReceipt(
                    "rolled_back",
                    plan.profile,
                    plan.digest,
                    plan.file_count,
                    plan.byte_count,
                    plan.preserved_raw_count,
                    plan.preserved_raw_bytes,
                    plan.preserved_raw_digest,
                    reason=type(exc).__name__,
                )

    def _remove_new_reset_state(self) -> None:
        for target in self._reset_targets():
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    def _reset_targets(self) -> tuple[Path, ...]:
        data = self.paths.data_root
        targets = [
            self.paths.lake_root / "silver",
            self.paths.lake_root / "gold",
            self.paths.lake_root / "metadata",
            data / "factors",
            data / "strategies",
            data / "todos",
            self.paths.experiments_root,
            self.paths.registries_root,
            data / "universes",
            self.paths.control_db_path,
            self.paths.sessions_root,
            self.paths.approvals_root,
            self.paths.order_plans_root,
            self.paths.reports_root,
            self.paths.audit_root,
            self.paths.quarantine_root,
            self.paths.backup_root,
            self.paths.project_root / "src/qmt_agent_trader/agent/generated",
        ]
        targets.extend(sorted(data.glob(f"{self.paths.control_db_path.name}.*")))
        return tuple(dict.fromkeys(path.resolve(strict=False) for path in targets))

    def _validate_reset_target(self, target: Path) -> None:
        root = self.paths.project_root
        if target != root and root not in target.parents:
            raise StorageValidationError(
                store_name="storage_reset",
                path=target,
                operation="plan_reset",
                reason="reset target escapes project root",
            )
        paths = [target]
        if target.is_dir():
            paths.extend(target.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise StorageValidationError(
                store_name="storage_reset",
                path=target,
                operation="plan_reset",
                reason="reset target contains a symbolic link",
            )

    def _snapshot_reset_files(
        self, root: Path, *, validate_parquet: bool = False
    ) -> list[tuple[str, int, str]]:
        if not root.exists():
            return []
        self._validate_reset_target(root)
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        snapshot: list[tuple[str, int, str]] = []
        for path in candidates:
            if not path.is_file():
                continue
            if validate_parquet and path.suffix != ".parquet":
                raise StorageValidationError(
                    store_name="storage_reset",
                    path=path,
                    operation="plan_reset",
                    reason="preserve-raw accepts only Parquet files",
                )
            if validate_parquet:
                try:
                    parquet = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
                    for index in range(parquet.num_row_groups):
                        parquet.read_row_group(index)  # type: ignore[no-untyped-call]
                except Exception as exc:
                    raise StorageValidationError(
                        store_name="storage_reset",
                        path=path,
                        operation="plan_reset",
                        reason="raw dataset is corrupt",
                        original_error=exc,
                    ) from exc
            snapshot.append(
                (
                    path.relative_to(self.paths.project_root).as_posix(),
                    path.stat().st_size,
                    _hash(path),
                )
            )
        return snapshot

    def verify(self, *, deep: bool = False) -> VerificationResult:
        diagnostics: list[StorageDiagnostic] = []
        if self.paths.control_db_path.exists():
            try:
                with self.database.read_connection("storage_verify", read_only=True) as connection:
                    connection.execute("SELECT 1").fetchone()
                    try:
                        rows = connection.execute(
                            "SELECT status FROM storage_schema_migrations WHERE status != 'APPLIED'"
                        ).fetchall()
                        if rows:
                            diagnostics.append(
                                StorageDiagnostic(
                                    "control_db",
                                    "MIGRATION_PENDING",
                                    "migration registry contains non-applied entries",
                                )
                            )
                    except Exception as exc:
                        if "does not exist" not in str(exc):
                            raise
            except StorageError as exc:
                diagnostics.append(
                    StorageDiagnostic(
                        "control_db",
                        _database_diagnostic_code(exc),
                        exc.reason,
                        self.paths.control_db_path,
                    )
                )
            try:
                pending = MigrationRegistry(self.database).apply(storage_migrations(), dry_run=True)
                if pending:
                    diagnostics.append(
                        StorageDiagnostic(
                            "control_db",
                            "MIGRATION_PENDING",
                            "pending immutable migrations: " + ", ".join(pending),
                            self.paths.control_db_path,
                        )
                    )
            except StorageConflictError as exc:
                diagnostics.append(
                    StorageDiagnostic(
                        "control_db",
                        "MIGRATION_CHECKSUM_MISMATCH",
                        exc.reason,
                        self.paths.control_db_path,
                    )
                )
        seen: set[Path] = set()
        for store in self.catalog.stores:
            root = store.path
            if not root.exists() or root in seen:
                continue
            seen.add(root)
            if store.governed and root.is_dir():
                for artifact_diagnostic in artifact_store_for_root(
                    root, lock_manager=self.locks
                ).diagnose():
                    diagnostics.append(
                        StorageDiagnostic(
                            store.name,
                            artifact_diagnostic.code,
                            artifact_diagnostic.reason,
                            root / artifact_diagnostic.relative_path,
                        )
                    )
                continue
            if store.verifier_id == "order_plan_event_stream_v1":
                diagnostics.extend(self._verify_order_plan_event_store(store))
                continue
            candidates = [root] if root.is_file() else root.rglob("*")
            for path in candidates:
                if not path.is_file() or self.paths.backup_root in path.parents:
                    continue
                diagnostics.extend(self._verify_store_file(store, path, deep=deep))
        return VerificationResult(
            not any(d.severity == "error" for d in diagnostics), deep, diagnostics
        )

    def _verify_order_plan_event_store(
        self, definition: StoreDefinition
    ) -> list[StorageDiagnostic]:
        artifact_store = artifact_store_for_root(
            self.paths.order_plans_root, lock_manager=self.locks
        )
        diagnostics: list[StorageDiagnostic] = []
        with self.locks.resource_lock(artifact_store._resource):
            if not definition.path.exists():
                return diagnostics
            for path in sorted(definition.path.glob("*.jsonl")):
                diagnostics.extend(
                    self._order_plan_event_diagnostics_assume_locked(
                        definition,
                        path,
                        artifact_store=artifact_store,
                    )
                )
        return diagnostics

    def _order_plan_event_diagnostics_assume_locked(
        self,
        definition: StoreDefinition,
        path: Path,
        *,
        artifact_store: ArtifactStore,
    ) -> list[StorageDiagnostic]:
        verification = verify_bound_order_plan_event_stream_assume_locked(
            store=artifact_store,
            path=path,
        )
        return [
            StorageDiagnostic(
                definition.name,
                item.code,
                item.reason,
                path,
            )
            for item in verification.corruptions
        ]

    def migrate(self, *, dry_run: bool) -> list[str]:
        return MigrationRegistry(self.database).apply(storage_migrations(), dry_run=dry_run)

    def backup(self) -> BackupReceipt:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        staging = self.paths.backup_root / f".{timestamp}-{uuid4().hex}.tmp"
        final = self.paths.backup_root / timestamp
        try:
            staging.mkdir(parents=True, exist_ok=False)
            files: list[dict[str, Any]] = []
            with self.locks.backup_barrier():
                for source in self._iter_backup_files():
                    relative = source.relative_to(self.paths.project_root)
                    target = staging / "files" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source == self.paths.control_db_path:
                        self.database.checkpoint_copy(target)
                    else:
                        shutil.copy2(source, target)
                    files.append(
                        {
                            "source": relative.as_posix(),
                            "sha256": _hash(target),
                            "size": target.stat().st_size,
                        }
                    )
            manifest = {
                "schema_version": 1,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "files": files,
                "scope": "local_consistent_v1",
            }
            self.atomic.write_json(staging / "manifest.json", manifest)
            verification = self._verify_backup(staging, require_success=False)
            if not verification.healthy:
                raise ValueError("backup hash verification failed")
            os.replace(staging, final)
            self.atomic.write_json(
                final / "SUCCESS.json",
                {"manifest_sha256": _hash(final / "manifest.json")},
                create_only=True,
            )
            return BackupReceipt(final, final / "manifest.json")
        except (StorageLockTimeoutError, StorageConflictError):
            shutil.rmtree(staging, ignore_errors=True)
            if final.exists() and not (final / "SUCCESS.json").exists():
                shutil.rmtree(final, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if final.exists() and not (final / "SUCCESS.json").exists():
                shutil.rmtree(final, ignore_errors=True)
            raise StorageBackupError(
                store_name="backups",
                path=final,
                operation="backup",
                reason="consistent local backup failed",
                recoverable=True,
                suggested_repair="inspect free space and storage locks, then retry",
                original_error=exc,
            ) from exc

    def verify_backup(self, root: Path) -> VerificationResult:
        return self._verify_backup(root, require_success=True)

    def _verify_backup(self, root: Path, *, require_success: bool) -> VerificationResult:
        diagnostics: list[StorageDiagnostic] = []
        try:
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
                raise ValueError("unsupported backup manifest schema")
            files_root = (root / "files").resolve()
            expected: set[Path] = set()
            sources: set[str] = set()
            for item in manifest["files"]:
                source = item.get("source")
                if not isinstance(source, str) or source in sources:
                    raise ValueError("invalid or duplicate backup source")
                sources.add(source)
                raw = Path(source)
                path = (files_root / raw).resolve()
                if raw.is_absolute() or files_root not in path.parents:
                    raise ValueError("backup source path escapes files root")
                expected.add(path)
                if (
                    not path.is_file()
                    or path.stat().st_size != item.get("size")
                    or _hash(path) != item.get("sha256")
                ):
                    diagnostics.append(
                        StorageDiagnostic(
                            "backup",
                            "HASH_MISMATCH",
                            "backup file missing or hash mismatched",
                            path,
                        )
                    )
            actual = {path.resolve() for path in files_root.rglob("*") if path.is_file()}
            for extra in sorted(actual - expected):
                diagnostics.append(
                    StorageDiagnostic("backup", "EXTRA_FILE", "unmanifested backup file", extra)
                )
            if require_success:
                success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
                if success.get("manifest_sha256") != _hash(manifest_path):
                    raise ValueError("success marker is not bound to manifest")
            snapshot = StorageOperations(self._rebased_paths(files_root))
            snapshot_result = snapshot.verify(deep=True)
            diagnostics.extend(snapshot_result.diagnostics)
        except Exception as exc:
            diagnostics.append(
                StorageDiagnostic(
                    "backup", "INVALID_MANIFEST", type(exc).__name__, root / "manifest.json"
                )
            )
        return VerificationResult(not diagnostics, True, diagnostics)

    def _rebased_paths(self, project_root: Path) -> PersistencePaths:
        values: dict[str, Path] = {"project_root": project_root}
        for name in PersistencePaths.__dataclass_fields__:
            if name == "project_root":
                continue
            original = getattr(self.paths, name)
            values[name] = project_root / original.relative_to(self.paths.project_root)
        return PersistencePaths(**values)

    def locks_report(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        known: dict[Path, tuple[tuple[str, ...], str]] = {}
        for store in self.catalog.stores:
            lock_path = self.locks.lock_path_for_resource(store.lock_resource)
            names, _ = known.get(lock_path, ((), store.lock_resource))
            known[lock_path] = ((*names, store.name), store.lock_resource)
        database_digest = hashlib.sha256(
            str(self.paths.control_db_path.resolve()).encode()
        ).hexdigest()
        known[self.paths.locks_root / f"database-{database_digest}.lock"] = (
            ("control_db",),
            str(self.paths.control_db_path.resolve()),
        )
        known[self.paths.locks_root / "writer-admission.lock"] = (
            ("writer_admission",),
            "writer-admission",
        )
        if not self.paths.locks_root.exists():
            return result
        for path in sorted(self.paths.locks_root.glob("*.lock")):
            active = _lock_is_active(path)
            known_metadata = known.get(path)
            result.append(
                {
                    "path": str(path),
                    "resource": (
                        known_metadata[1] if known_metadata is not None else path.stem
                    ),
                    "known_resources": (
                        known_metadata[0] if known_metadata is not None else ()
                    ),
                    "resource_status": (
                        "known" if known_metadata is not None else "unknown"
                    ),
                    "active": active,
                }
            )
        maintenance = self.paths.locks_root / "maintenance.active"
        if maintenance.exists():
            result.append(_marker_report(maintenance, marker_type="maintenance"))
        writers_root = self.paths.locks_root / "writers"
        result.extend(
            _marker_report(path, marker_type="writer")
            for path in sorted(writers_root.glob("*.json"))
        )
        return result

    def quarantine(self, store: str, record: str) -> QuarantineReceipt:
        try:
            definition = self.catalog.by_name(store)
        except KeyError:
            raise StorageValidationError(
                store_name="quarantine",
                path=self.paths.quarantine_root,
                operation="resolve",
                reason="unknown store",
            ) from None
        raw = Path(record)
        is_single_file = definition.layout == "single_file"
        root = definition.path.parent if is_single_file else definition.path
        source = (root / raw).resolve()
        expected_file = definition.path.resolve() if is_single_file else None
        invalid_single_file = expected_file is not None and source != expected_file
        if (
            raw.is_absolute()
            or invalid_single_file
            or root.resolve() not in source.parents
            or (not definition.governed and not source.is_file())
        ):
            raise StorageValidationError(
                store_name="quarantine",
                path=source,
                operation="resolve",
                reason="record path is unsafe or missing",
            )
        if definition.governed:
            artifact_store = artifact_store_for_root(root, lock_manager=self.locks)
            relative = source.relative_to(root).as_posix()
            auxiliary_factory = None
            if definition.name == "order_plans":

                def order_plan_auxiliary(artifact_id: str) -> tuple[Path, ...]:
                    event_path = (
                        root
                        / ".events"
                        / (hashlib.sha256(artifact_id.encode()).hexdigest() + ".jsonl")
                    )
                    return (event_path,) if event_path.is_file() else ()

                auxiliary_factory = order_plan_auxiliary
            receipt = artifact_store.quarantine_relative_path(
                relative,
                quarantine_root=self.paths.quarantine_root / store,
                auxiliary_paths=auxiliary_factory,
            )
            if receipt is not None:
                primary = (
                    receipt.quarantined_content_path
                    or receipt.quarantined_manifest_path
                )
                return QuarantineReceipt(primary, receipt.sidecar_path)
        target_root = self.paths.quarantine_root / store
        target = (
            target_root
            / f"{source.name}.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S.%fZ')}.quarantine"
        )
        manifest_path = target.with_suffix(target.suffix + ".json")
        event_artifact_store = (
            artifact_store_for_root(
                self.paths.order_plans_root,
                lock_manager=self.locks,
            )
            if definition.verifier_id == "order_plan_event_stream_v1"
            else None
        )
        quarantine_lock: str | Path = (
            event_artifact_store._resource
            if event_artifact_store is not None
            else source
        )
        with self.locks.resource_lock(quarantine_lock):
            validation = (
                self._order_plan_event_diagnostics_assume_locked(
                    definition,
                    source,
                    artifact_store=event_artifact_store,
                )
                if event_artifact_store is not None
                else self._verify_store_file(definition, source, deep=True)
            )
            if not any(item.severity == "error" for item in validation):
                raise StorageValidationError(
                    store_name="quarantine",
                    path=source,
                    operation="quarantine",
                    reason="authoritative record is valid; quarantine is corruption-only",
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = _hash(source), source.stat().st_size
            os.replace(source, target)
            try:
                write_sidecar = (
                    self.atomic.write_json_assume_locked
                    if definition.verifier_id == "order_plan_event_stream_v1"
                    else self.atomic.write_json
                )
                write_sidecar(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "store": store,
                        "original_path": str(source),
                        "quarantine_path": str(target),
                        "sha256": digest,
                        "size": size,
                        "reason": "explicit operator quarantine",
                        **(
                            {
                                "diagnostics": [
                                    {
                                        "component": item.component,
                                        "code": item.code,
                                        "reason": item.reason,
                                        "path": str(item.path) if item.path else None,
                                        "severity": item.severity,
                                    }
                                    for item in validation
                                ]
                            }
                            if event_artifact_store is not None
                            else {}
                        ),
                        "quarantined_at": datetime.now(tz=UTC).isoformat(),
                    },
                    create_only=True,
                )
            except Exception:
                if target.exists() and not source.exists():
                    os.replace(target, source)
                manifest_path.unlink(missing_ok=True)
                raise
        return QuarantineReceipt(target, manifest_path)

    def _iter_backup_files(self) -> Iterator[Path]:
        seen: set[Path] = set()
        for store in self.catalog.stores:
            root = store.path
            candidates = [root] if root.is_file() else root.rglob("*") if root.exists() else []
            for path in candidates:
                if (
                    path.is_file()
                    and path not in seen
                    and self.paths.cache_root not in (path, *path.parents)
                    and self.paths.locks_root not in (path, *path.parents)
                    and self.paths.backup_root not in (path, *path.parents)
                    and not path.name.endswith((".tmp", ".lock"))
                ):
                    seen.add(path)
                    yield path

    def _verify_file(self, path: Path, *, deep: bool) -> list[StorageDiagnostic]:
        result: list[StorageDiagnostic] = []
        try:
            if path.suffix == ".parquet":
                parquet = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
                if deep:
                    for index in range(parquet.num_row_groups):
                        parquet.read_row_group(index)  # type: ignore[no-untyped-call]
            elif path.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and "content_hash" in value and "data" in value:
                    actual = hashlib.sha256(
                        json.dumps(value["data"], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    if value["content_hash"] != actual:
                        result.append(
                            StorageDiagnostic(
                                "json", "HASH_MISMATCH", "content hash mismatch", path
                            )
                        )
            elif path.suffix in {".yaml", ".yml"}:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            elif path.suffix == ".jsonl":
                raw = path.read_bytes()
                for line in raw.splitlines():
                    json.loads(line)
                if raw and not raw.endswith(b"\n"):
                    result.append(
                        StorageDiagnostic(
                            "jsonl",
                            "INCOMPLETE_TAIL",
                            "incomplete final JSONL record",
                            path,
                            "warning",
                        )
                    )
            if path.name.endswith(".tmp"):
                result.append(
                    StorageDiagnostic(
                        "filesystem", "STALE_TEMP", "temporary artifact remains", path, "warning"
                    )
                )
        except Exception as exc:
            code = "PARQUET_CORRUPT" if path.suffix == ".parquet" else "INVALID_CONTENT"
            result.append(StorageDiagnostic("file", code, type(exc).__name__, path))
        return result

    def _verify_store_file(
        self, store: StoreDefinition, path: Path, *, deep: bool
    ) -> list[StorageDiagnostic]:
        if store.name == "audit" and path.suffix == ".jsonl":
            audit_verification = AuditJsonlStore(path, self.atomic).verify()
            diagnostics = [
                StorageDiagnostic(
                    store.name,
                    "INVALID_CONTENT",
                    item.reason,
                    path,
                )
                for item in audit_verification.corruptions
            ]
            if audit_verification.tail_truncated:
                diagnostics.append(
                    StorageDiagnostic(
                        store.name,
                        "INCOMPLETE_TAIL",
                        "incomplete final JSONL record",
                        path,
                    )
                )
            return diagnostics
        if store.governed:
            root = store.path
            relative = path.relative_to(root).as_posix()
            matching = [
                item
                for item in artifact_store_for_root(root, lock_manager=self.locks).diagnose()
                if item.relative_path == relative
            ]
            if not matching:
                return []
            return [
                StorageDiagnostic(store.name, item.code, item.reason, path) for item in matching
            ]
        if store.verifier_id.startswith("versioned_registry_"):
            return _verify_versioned_json(path, store, record_kind=None)
        if store.verifier_id.startswith("versioned_record_"):
            record_kind = store.verifier_id.removeprefix("versioned_record_").removesuffix("_v2")
            return _verify_versioned_json(path, store, record_kind=record_kind)
        return [
            StorageDiagnostic(store.name, item.code, item.reason, item.path, item.severity)
            for item in self._verify_file(path, deep=deep)
        ]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_diagnostic_code(error: StorageError) -> str:
    if isinstance(error, StorageLockTimeoutError):
        return "CONTROL_DB_LOCKED"
    if isinstance(error, StoragePermissionError):
        return "CONTROL_DB_PERMISSION_DENIED"
    if isinstance(error, StorageSchemaMismatchError):
        return "CONTROL_DB_SCHEMA_MISSING"
    if isinstance(error, StorageCorruptError):
        return "CONTROL_DB_CORRUPT"
    if isinstance(error, StorageUnavailableError):
        return "CONTROL_DB_UNAVAILABLE"
    return "CONTROL_DB_ERROR"


def _lock_is_active(path: Path) -> bool:
    from filelock import FileLock, Timeout

    lock = FileLock(str(path), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return True
    else:
        lock.release()
        return False


def _marker_report(path: Path, *, marker_type: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        host = str(payload["host"])
        active = host != socket.gethostname() or _pid_is_alive(pid)
        return {
            "path": str(path),
            "resource": payload.get("resource"),
            "known_resources": (),
            "resource_status": marker_type,
            "active": active,
            "pid": pid,
            "operation": payload.get("operation"),
            "started_at": payload.get("started_at"),
            "host": host,
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "resource": None,
            "known_resources": (),
            "resource_status": f"invalid_{marker_type}_marker",
            "active": None,
            "error": type(exc).__name__,
        }


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _verify_versioned_json(
    path: Path, store: StoreDefinition, *, record_kind: str | None
) -> list[StorageDiagnostic]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("root must be an object")
        if record_kind is None:
            _registry_validator(store, path).validate_payload(payload)
        else:
            _record_validator(store, record_kind, path).validate_payload(
                payload, record_id=path.stem, path=path
            )
    except StorageSchemaMismatchError as exc:
        return [StorageDiagnostic(store.name, "SCHEMA_MISMATCH", exc.reason, path)]
    except StorageCorruptError as exc:
        return [StorageDiagnostic(store.name, "HASH_MISMATCH", exc.reason, path)]
    except StorageValidationError as exc:
        code = "IDENTITY_MISMATCH" if "identity" in exc.reason else "INVALID_CONTENT"
        return [StorageDiagnostic(store.name, code, exc.reason, path)]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return [StorageDiagnostic(store.name, "INVALID_CONTENT", type(exc).__name__, path)]
    return []


def _registry_validator(store: StoreDefinition, path: Path) -> Any:
    from qmt_agent_trader.persistence.repositories.versioned_json import VersionedJsonRegistry

    if store.name == "factor_registry":
        from qmt_agent_trader.factors.registry import SavedFactor, _load_file_factor

        loader: Any = _load_file_factor
        dumper: Any = SavedFactor.to_dict

        def identity(item: Any) -> str:
            return str(item.factor_id)
    else:
        from qmt_agent_trader.strategy.registry import _load_file_strategy

        loader = _load_file_strategy

        def dumper(item: Any) -> dict[str, Any]:
            return dict(item.model_dump(mode="json"))

        def identity(item: Any) -> str:
            return str(item.strategy_id)

    manager = LockManager(path.parent / ".verify-locks")
    return VersionedJsonRegistry(
        path=path,
        item_loader=loader,
        item_dumper=dumper,
        item_identity=identity,
        lock_manager=manager,
        atomic_store=AtomicFileStore(manager),
        store_name=store.name,
    )


def _record_validator(store: StoreDefinition, record_kind: str, path: Path) -> Any:
    from qmt_agent_trader.persistence.repositories.versioned_record import (
        VersionedRecordRepository,
    )

    identity: Any = None
    model: Any
    if record_kind == "todos":
        from qmt_agent_trader.agent.todos import TodoListRecord

        model = TodoListRecord

        def identity(record: Any) -> str:
            return hashlib.sha256(record.session_id.encode()).hexdigest()[:16]
    elif record_kind == "experiments":
        from qmt_agent_trader.agent.schemas import ExperimentRecord

        model = ExperimentRecord

        def identity(record: Any) -> str:
            return str(record.experiment_id)
    elif record_kind == "sessions":
        from qmt_agent_trader.web.schemas import ChatSession

        model = ChatSession

        def identity(record: Any) -> str:
            return str(record.session_id)
    else:
        from qmt_agent_trader.universe.registry import UniverseStoredRecord

        model = UniverseStoredRecord

        def identity(record: Any) -> str:
            return str(record.spec.universe_id)

    return VersionedRecordRepository(
        path.parent,
        model,
        store_name=store.name,
        locks_root=path.parent / ".verify-locks",
        identity=identity,
    )


def as_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
