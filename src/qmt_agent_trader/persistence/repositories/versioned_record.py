"""Versioned one-record-per-file JSON repositories."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import (
    StorageCorruptError,
    StorageError,
    StorageRevisionConflictError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class RecordDiagnostic:
    path: Path
    error: StorageError


class VersionedRecordRepository(Generic[T]):
    """Canonical locked RMW protocol for mutable JSON records."""

    def __init__(
        self,
        root: Path,
        model: type[T],
        *,
        store_name: str,
        locks_root: Path | None = None,
        quarantine_root: Path | None = None,
        identity: Callable[[T], str] | None = None,
        fault_hook: Callable[[str, Path], None] | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.model = model
        self.store_name = store_name
        self.identity = identity
        self.fault_hook = fault_hook
        locks = (locks_root or self.root.parent / "locks").expanduser().resolve()
        self.quarantine_root = (
            (quarantine_root or self.root.parent / "quarantine" / store_name).expanduser().resolve()
        )
        self.lock_manager = LockManager(locks)
        self.atomic_store = AtomicFileStore(self.lock_manager)

    def path_for(self, record_id: str) -> Path:
        if not record_id or record_id in {".", ".."} or any(c in record_id for c in "/\\\0"):
            raise StorageValidationError(
                store_name=self.store_name,
                path=self.root,
                operation="path_for",
                reason="record id is not path-safe",
            )
        return self.root / f"{record_id}.json"

    def load(self, record_id: str, *, missing: Callable[[], T] | None = None) -> T:
        path = self.path_for(record_id)
        with self.lock_manager.resource_lock(path):
            if not path.exists():
                if missing is not None:
                    return missing()
                raise FileNotFoundError(path)
            record = self._load_locked(path, migrate=True)
            self._validate_identity(record_id, record, path)
            return record

    def create(self, record_id: str, record: T) -> T:
        path = self.path_for(record_id)
        with self.lock_manager.resource_lock(path):
            if path.exists():
                raise StorageRevisionConflictError(
                    store_name=self.store_name,
                    path=path,
                    operation="create",
                    reason="record already exists",
                )
            self._validate_identity(record_id, record, path)
            return self._write_locked(path, record, revision=1)

    def mutate(
        self,
        record_id: str,
        operation: Callable[[T], T],
        *,
        missing: Callable[[], T] | None = None,
        expected_revision: int | None = None,
    ) -> T:
        path = self.path_for(record_id)
        with self.lock_manager.resource_lock(path):
            current = (
                missing()
                if not path.exists() and missing
                else self._load_locked(path, migrate=True)
            )
            revision = int(getattr(current, "revision", 0))
            if expected_revision is not None and expected_revision != revision:
                raise StorageRevisionConflictError(
                    store_name=self.store_name,
                    path=path,
                    operation="mutate",
                    reason=f"expected revision {expected_revision}, found {revision}",
                )
            next_record = operation(current)
            self._validate_identity(record_id, next_record, path)
            return self._write_locked(path, next_record, revision=revision + 1)

    def upsert(
        self,
        record_id: str,
        operation: Callable[[T | None], T],
        *,
        expected_revision: int | None = None,
    ) -> T:
        path = self.path_for(record_id)
        with self.lock_manager.resource_lock(path):
            current = self._load_locked(path, migrate=True) if path.exists() else None
            revision = int(getattr(current, "revision", 0)) if current is not None else 0
            if expected_revision is not None and expected_revision != revision:
                raise StorageRevisionConflictError(
                    store_name=self.store_name,
                    path=path,
                    operation="upsert",
                    reason=f"expected revision {expected_revision}, found {revision}",
                )
            next_record = operation(current)
            self._validate_identity(record_id, next_record, path)
            return self._write_locked(path, next_record, revision=revision + 1)

    def list_with_diagnostics(self) -> tuple[list[T], list[RecordDiagnostic]]:
        records: list[T] = []
        diagnostics: list[RecordDiagnostic] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                records.append(self.load(path.stem))
            except (StorageCorruptError, StorageValidationError) as exc:
                diagnostics.append(RecordDiagnostic(path, exc))
        return records, diagnostics

    def quarantine(self, record_id: str) -> Path:
        path = self.path_for(record_id)
        with self.lock_manager.resource_lock(path):
            # Validate first: quarantine is explicit but only applies to bad records.
            try:
                self._load_locked(path, migrate=False)
            except (StorageCorruptError, StorageValidationError):
                pass
            else:
                raise StorageValidationError(
                    store_name=self.store_name,
                    path=path,
                    operation="quarantine",
                    reason="record is valid",
                )
            self.quarantine_root.mkdir(parents=True, exist_ok=True)
            target = (
                self.quarantine_root / f"{path.stem}-{shanghai_now_iso().replace(':', '')}.json"
            )
            os.replace(path, target)
            return target

    def delete(self, record_id: str) -> bool:
        path = self.path_for(record_id)
        with self.lock_manager.resource_lock(path):
            if not path.exists():
                return False
            path.unlink()
            return True

    def _load_locked(self, path: Path, *, migrate: bool) -> T:
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StorageCorruptError(
                store_name=self.store_name,
                path=path,
                operation="load",
                reason="record JSON is unreadable",
                original_error=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise StorageValidationError(
                store_name=self.store_name,
                path=path,
                operation="load",
                reason="record root must be an object",
            )
        if "schema_version" not in payload:
            if not migrate:
                raise StorageValidationError(
                    store_name=self.store_name,
                    path=path,
                    operation="load",
                    reason="legacy record has no schema version",
                )
            try:
                legacy = self.model.model_validate(payload)
            except ValidationError as exc:
                raise StorageValidationError(
                    store_name=self.store_name,
                    path=path,
                    operation="migrate",
                    reason="legacy record is invalid",
                    original_error=exc,
                ) from exc
            self._validate_identity(path.stem, legacy, path)
            return self._write_locked(path, legacy, revision=1)
        raw_hash = payload.pop("content_hash", None)
        expected = self._hash(payload)
        if raw_hash != expected:
            raise StorageCorruptError(
                store_name=self.store_name,
                path=path,
                operation="validate_hash",
                reason="record content hash mismatch",
            )
        try:
            return self.model.model_validate(payload)
        except ValidationError as exc:
            raise StorageValidationError(
                store_name=self.store_name,
                path=path,
                operation="load",
                reason="record schema validation failed",
                original_error=exc,
            ) from exc

    def _write_locked(self, path: Path, record: T, *, revision: int) -> T:
        payload = record.model_dump(mode="json")
        payload.update(schema_version=2, revision=revision, updated_at=shanghai_now_iso())
        validated = self.model.model_validate(payload)
        canonical = validated.model_dump(mode="json")
        disk = {**canonical, "content_hash": self._hash(canonical)}
        self.atomic_store.write_json(path, disk, fault_hook=self.fault_hook)
        return self._load_locked(path, migrate=False)

    def _validate_identity(self, record_id: str, record: T, path: Path) -> None:
        if self.identity is not None and self.identity(record) != record_id:
            raise StorageValidationError(
                store_name=self.store_name,
                path=path,
                operation="validate_identity",
                reason="record identity does not match storage key",
            )

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()
