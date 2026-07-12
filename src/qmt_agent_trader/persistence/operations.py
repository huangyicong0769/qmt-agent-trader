"""Local storage inventory, health, migration, backup, and quarantine operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pyarrow.parquet as pq
import yaml

from qmt_agent_trader.persistence.artifacts import artifact_store_for_root
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.catalog import StoreCatalog, StoreDefinition
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


@dataclass(frozen=True)
class StoreInventory:
    name: str
    type: str
    path: Path
    owner: str
    source_of_truth: str
    schema_version: int | None
    mutable: bool
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
                lock_policy=store.lock_resource,
                backup_policy=store.backup,
                health="present" if store.path.exists() else "not_initialized",
            )
            for store in self.catalog.stores
        ]

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
            candidates = [root] if root.is_file() else root.rglob("*")
            for path in candidates:
                if not path.is_file() or self.paths.backup_root in path.parents:
                    continue
                diagnostics.extend(self._verify_store_file(store, path, deep=deep))
        return VerificationResult(
            not any(d.severity == "error" for d in diagnostics), deep, diagnostics
        )

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
        now = datetime.now(tz=UTC).timestamp()
        result: list[dict[str, Any]] = []
        known = {
            self.locks.lock_path_for_resource(store.lock_resource): store.name
            for store in self.catalog.stores
        }
        database_digest = hashlib.sha256(
            str(self.paths.control_db_path.resolve()).encode()
        ).hexdigest()
        known[self.paths.locks_root / f"database-{database_digest}.lock"] = "control_db"
        known[self.paths.locks_root / "backup-barrier.lock"] = "backup_barrier"
        if not self.paths.locks_root.exists():
            return result
        for path in sorted(self.paths.locks_root.glob("*.lock")):
            age = max(0.0, now - path.stat().st_mtime)
            active = _lock_is_active(path)
            resource = known.get(path)
            result.append(
                {
                    "path": str(path),
                    "resource": path.stem,
                    "known_resource": resource,
                    "resource_status": "known" if resource is not None else "unknown",
                    "age_seconds": age,
                    "stale": age > 3600 and not active,
                    "stale_basis": "mtime_only_no_owner_evidence",
                    "active": active,
                }
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
        root = definition.path.parent if definition.path.is_file() else definition.path
        source = (root / raw).resolve()
        if raw.is_absolute() or root.resolve() not in source.parents or not source.is_file():
            raise StorageValidationError(
                store_name="quarantine",
                path=source,
                operation="resolve",
                reason="record path is unsafe or missing",
            )
        target_root = self.paths.quarantine_root / store
        target = (
            target_root
            / f"{source.name}.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S.%fZ')}.quarantine"
        )
        manifest_path = target.with_suffix(target.suffix + ".json")
        with self.locks.resource_lock(source):
            validation = self._verify_store_file(definition, source, deep=True)
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
                self.atomic.write_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "store": store,
                        "original_path": str(source),
                        "quarantine_path": str(target),
                        "sha256": digest,
                        "size": size,
                        "reason": "explicit operator quarantine",
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


def _verify_versioned_json(
    path: Path, store: StoreDefinition, *, record_kind: str | None
) -> list[StorageDiagnostic]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("root must be an object")
        if payload.get("schema_version") != 2:
            return [
                StorageDiagnostic(store.name, "SCHEMA_MISMATCH", "schema_version must be 2", path)
            ]
        revision = payload.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            return [
                StorageDiagnostic(
                    store.name, "REVISION_INVALID", "revision must be non-negative", path
                )
            ]
        claimed = payload.get("content_hash")
        unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
        if record_kind is None:
            expected = hashlib.sha256(
                json.dumps(
                    unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
        else:
            expected = hashlib.sha256(
                json.dumps(
                    unhashed, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        if claimed != expected:
            return [StorageDiagnostic(store.name, "HASH_MISMATCH", "content hash mismatch", path)]
        identity = _record_identity(record_kind, payload)
        if identity is not None and identity != path.stem:
            return [
                StorageDiagnostic(
                    store.name, "IDENTITY_MISMATCH", "record identity does not match filename", path
                )
            ]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        return [StorageDiagnostic(store.name, "INVALID_CONTENT", type(exc).__name__, path)]
    return []


def _record_identity(record_kind: str | None, payload: dict[str, Any]) -> str | None:
    if record_kind == "sessions":
        return str(payload["session_id"])
    if record_kind == "experiments":
        return str(payload["experiment_id"])
    if record_kind == "universes":
        return str(payload["spec"]["universe_id"])
    if record_kind == "todos":
        return hashlib.sha256(str(payload["session_id"]).encode("utf-8")).hexdigest()[:16]
    return None


def as_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
