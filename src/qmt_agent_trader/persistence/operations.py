"""Local storage inventory, health, migration, backup, and quarantine operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pyarrow.parquet as pq
import yaml

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import StorageBackupError, StorageValidationError
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

    def inventory(self) -> list[StoreInventory]:
        result: list[StoreInventory] = []
        for name in PersistencePaths.__dataclass_fields__:
            if name == "project_root":
                continue
            path = getattr(self.paths, name)
            is_db = name == "control_db_path"
            is_cache = name == "cache_root"
            is_infra = name in {"locks_root", "quarantine_root", "backup_root"}
            result.append(
                StoreInventory(
                    name=name,
                    type="duckdb" if is_db else "directory",
                    path=path,
                    owner="persistence" if is_infra or is_db else "application",
                    source_of_truth="cache" if is_cache else ("catalog" if is_db else "official"),
                    schema_version=1 if is_db else None,
                    mutable=not name.endswith("artifact_root"),
                    lock_policy="database write lock" if is_db else "canonical resource lock",
                    backup_policy="excluded"
                    if is_cache or name in {"locks_root", "backup_root"}
                    else "local consistent backup v1",
                    health="present" if path.exists() else "not_initialized",
                )
            )
        return result

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
            except Exception as exc:
                diagnostics.append(
                    StorageDiagnostic(
                        "control_db",
                        "DUCKDB_CORRUPT",
                        type(exc).__name__,
                        self.paths.control_db_path,
                    )
                )
        roots = self._official_roots(include_quarantine=True)
        seen: set[Path] = set()
        for root in roots:
            if not root.exists() or root in seen:
                continue
            seen.add(root)
            for path in root.rglob("*"):
                if not path.is_file() or self.paths.backup_root in path.parents:
                    continue
                diagnostics.extend(self._verify_file(path, deep=deep))
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
            resources = sorted(str(root) for root in self._official_roots(include_quarantine=True))
            with ExitStack() as stack:
                for resource in resources:
                    stack.enter_context(self.locks.resource_lock(f"backup:{resource}"))
                for source in self._iter_backup_files():
                    relative = source.relative_to(self.paths.project_root)
                    target = staging / "files" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
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
            verification = self.verify_backup(staging)
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
        diagnostics: list[StorageDiagnostic] = []
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            for item in manifest["files"]:
                path = root / "files" / item["source"]
                if not path.is_file() or _hash(path) != item["sha256"]:
                    diagnostics.append(
                        StorageDiagnostic(
                            "backup",
                            "HASH_MISMATCH",
                            "backup file missing or hash mismatched",
                            path,
                        )
                    )
        except Exception as exc:
            diagnostics.append(
                StorageDiagnostic(
                    "backup", "INVALID_MANIFEST", type(exc).__name__, root / "manifest.json"
                )
            )
        return VerificationResult(not diagnostics, True, diagnostics)

    def locks_report(self) -> list[dict[str, Any]]:
        now = datetime.now(tz=UTC).timestamp()
        result = []
        if not self.paths.locks_root.exists():
            return result
        for path in sorted(self.paths.locks_root.glob("*.lock")):
            age = max(0.0, now - path.stat().st_mtime)
            result.append(
                {
                    "path": str(path),
                    "resource": path.stem,
                    "age_seconds": age,
                    "stale": age > 3600,
                    "active": _lock_is_active(path),
                }
            )
        return result

    def quarantine(self, store: str, record: str) -> QuarantineReceipt:
        roots = {
            "sessions": self.paths.sessions_root,
            "experiments": self.paths.experiments_root,
            "registries": self.paths.registries_root,
            "artifacts": self.paths.artifact_root,
            "lake": self.paths.lake_root,
        }
        if store not in roots:
            raise StorageValidationError(
                store_name="quarantine",
                path=self.paths.quarantine_root,
                operation="resolve",
                reason="unknown store",
            )
        raw = Path(record)
        source = (roots[store] / raw).resolve()
        if (
            raw.is_absolute()
            or roots[store].resolve() not in source.parents
            or not source.is_file()
        ):
            raise StorageValidationError(
                store_name="quarantine",
                path=source,
                operation="resolve",
                reason="record path is unsafe or missing",
            )
        if source.suffix in {".json", ".yaml", ".yml"}:
            try:
                json.loads(source.read_text()) if source.suffix == ".json" else yaml.safe_load(
                    source.read_text()
                )
            except Exception:
                pass
            else:
                raise StorageValidationError(
                    store_name="quarantine",
                    path=source,
                    operation="quarantine",
                    reason="authoritative record is valid; quarantine is corruption-only",
                )
        target_root = self.paths.quarantine_root / store
        target = (
            target_root
            / f"{source.name}.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S.%fZ')}.quarantine"
        )
        manifest_path = target.with_suffix(target.suffix + ".json")
        with self.locks.resource_lock(source):
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = _hash(source), source.stat().st_size
            os.replace(source, target)
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
        return QuarantineReceipt(target, manifest_path)

    def health_payload(
        self,
        *,
        component: str,
        reason: str = "",
        warnings: list[str] | None = None,
        repair_action: str | None = None,
    ) -> dict[str, Any]:
        safe_reason = re.sub(r"(?i)(token|secret|password|key)\s*=\s*\S+", r"\1=[REDACTED]", reason)
        return {
            "storage_status": "degraded" if reason or warnings else "ok",
            "storage_component": component,
            "storage_reason": safe_reason,
            "storage_warnings": warnings or [],
            "storage_repair_action": repair_action,
        }

    def _official_roots(self, *, include_quarantine: bool) -> list[Path]:
        roots = [
            self.paths.lake_root,
            self.paths.control_db_path,
            self.paths.artifact_root,
            self.paths.reports_root,
            self.paths.approvals_root,
            self.paths.order_plans_root,
            self.paths.sessions_root,
            self.paths.experiments_root,
            self.paths.registries_root,
            self.paths.audit_root,
        ]
        if include_quarantine:
            roots.append(self.paths.quarantine_root)
        return roots

    def _iter_backup_files(self):
        seen: set[Path] = set()
        for root in self._official_roots(include_quarantine=True):
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
                parquet = pq.ParquetFile(path)
                if deep:
                    for index in range(parquet.num_row_groups):
                        parquet.read_row_group(index)
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


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def as_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
