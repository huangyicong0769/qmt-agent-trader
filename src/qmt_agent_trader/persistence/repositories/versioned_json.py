"""Locked, validated, versioned JSON registry snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import (
    StorageCorruptError,
    StorageRevisionConflictError,
    StorageSchemaMismatchError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager

T = TypeVar("T")
Mutation = Callable[[list[T]], Iterable[T]]

_V2_FIELDS = {"schema_version", "revision", "updated_at", "content_hash", "items"}


@dataclass(frozen=True)
class RegistrySnapshot(Generic[T]):
    schema_version: int
    revision: int
    updated_at: str
    content_hash: str
    items: tuple[T, ...]


class VersionedJsonRegistry(Generic[T]):
    """A JSON snapshot repository with locked read-modify-write semantics."""

    def __init__(
        self,
        *,
        path: Path,
        item_loader: Callable[[dict[str, Any]], T],
        item_dumper: Callable[[T], dict[str, Any]],
        item_identity: Callable[[T], str],
        lock_manager: LockManager,
        atomic_store: AtomicFileStore,
        store_name: str,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.item_loader = item_loader
        self.item_dumper = item_dumper
        self.item_identity = item_identity
        self.lock_manager = lock_manager
        self.atomic_store = atomic_store
        self.store_name = store_name

    def load_snapshot(self) -> RegistrySnapshot[T]:
        """Load and validate the latest current-schema snapshot without mutation."""
        with self.lock_manager.resource_lock(self.path):
            return self._load_locked()

    def mutate(
        self,
        operation: Mutation[T],
        *,
        expected_revision: int | None = None,
    ) -> RegistrySnapshot[T]:
        """Apply ``operation`` to the latest data and atomically persist revision+1."""
        with self.lock_manager.resource_lock(self.path):
            current = self._load_locked()
            if expected_revision is not None and expected_revision != current.revision:
                raise StorageRevisionConflictError(
                    store_name=self.store_name,
                    path=self.path,
                    operation="mutate",
                    reason=(f"expected revision {expected_revision}, found {current.revision}"),
                    recoverable=True,
                    suggested_repair="reload the latest snapshot and retry the mutation",
                )
            next_items = tuple(operation(list(current.items)))
            dumped = self._dump_and_validate_items(next_items, operation="mutate")
            current_dumped = self._dump_and_validate_items(
                current.items,
                operation="mutate",
            )
            if dumped == current_dumped:
                return current
            payload = self._payload(
                revision=current.revision + 1,
                updated_at=shanghai_now_iso(),
                items=dumped,
            )
            self._write_payload(payload)
            verified = self._load_locked()
            if (
                verified.revision != current.revision + 1
                or verified.content_hash != payload["content_hash"]
            ):
                raise StorageCorruptError(
                    store_name=self.store_name,
                    path=self.path,
                    operation="verify_write",
                    reason="written registry snapshot did not verify",
                    recoverable=False,
                    suggested_repair="restore the last verified registry backup",
                )
            return verified

    def _load_locked(self) -> RegistrySnapshot[T]:
        if not self.path.exists():
            return self._empty_snapshot()
        try:
            raw = self.path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StorageCorruptError(
                store_name=self.store_name,
                path=self.path,
                operation="load",
                reason="registry JSON is unreadable",
                recoverable=False,
                suggested_repair="quarantine the registry and restore a verified backup",
                original_error=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise self._schema_error("registry root must be an object")
        if payload.get("schema_version") == 2:
            return self._parse_v2(cast(dict[str, Any], payload))
        version = payload.get("schema_version", payload.get("version", "missing"))
        raise self._schema_error(f"unsupported registry schema version: {version}")

    def _parse_v2(self, payload: dict[str, Any]) -> RegistrySnapshot[T]:
        return self.validate_payload(payload)

    def validate_payload(self, payload: dict[str, Any]) -> RegistrySnapshot[T]:
        """Validate an already-decoded payload with the exact runtime rules."""
        if set(payload) != _V2_FIELDS:
            raise self._schema_error("v2 registry fields do not match the schema")
        revision = payload["revision"]
        updated_at = payload["updated_at"]
        content_hash = payload["content_hash"]
        raw_items = payload["items"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise self._schema_error("registry revision must be a non-negative integer")
        if not isinstance(updated_at, str) or not updated_at:
            raise self._schema_error("registry updated_at must be a non-empty string")
        if not isinstance(content_hash, str) or len(content_hash) != 64:
            raise self._schema_error("registry content_hash must be a SHA-256 digest")
        if not isinstance(raw_items, list):
            raise self._schema_error("registry items must be a list")
        expected_hash = _content_hash(
            schema_version=2,
            revision=revision,
            updated_at=updated_at,
            items=raw_items,
        )
        if content_hash != expected_hash:
            raise StorageCorruptError(
                store_name=self.store_name,
                path=self.path,
                operation="validate_hash",
                reason="registry content hash does not match its payload",
                recoverable=False,
                suggested_repair="restore a verified registry snapshot",
            )
        items = self._load_items(raw_items, operation="load")
        normalized = self._dump_and_validate_items(items, operation="load")
        if normalized != raw_items:
            raise StorageValidationError(
                store_name=self.store_name,
                path=self.path,
                operation="load",
                reason="registry item encoding is not canonical",
            )
        return RegistrySnapshot(
            schema_version=2,
            revision=revision,
            updated_at=updated_at,
            content_hash=content_hash,
            items=items,
        )

    def _empty_snapshot(self) -> RegistrySnapshot[T]:
        payload = self._payload(revision=0, updated_at="", items=[])
        return RegistrySnapshot(
            schema_version=2,
            revision=0,
            updated_at="",
            content_hash=cast(str, payload["content_hash"]),
            items=(),
        )

    def _load_items(self, raw_items: list[Any], *, operation: str) -> tuple[T, ...]:
        loaded: list[T] = []
        try:
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    raise TypeError("registry item must be an object")
                loaded.append(self.item_loader(cast(dict[str, Any], raw_item)))
        except Exception as exc:
            raise StorageValidationError(
                store_name=self.store_name,
                path=self.path,
                operation=operation,
                reason="registry item validation failed",
                original_error=exc,
            ) from exc
        return tuple(loaded)

    def _dump_and_validate_items(
        self,
        items: Iterable[T],
        *,
        operation: str,
    ) -> list[dict[str, Any]]:
        dumped: list[dict[str, Any]] = []
        identities: set[str] = set()
        try:
            for item in items:
                identity = self.item_identity(item)
                if not identity:
                    raise ValueError("registry item identity cannot be empty")
                if identity in identities:
                    raise ValueError(f"duplicate registry item identity: {identity}")
                identities.add(identity)
                raw = self.item_dumper(item)
                if not isinstance(raw, dict):
                    raise TypeError("registry item dumper must return an object")
                dumped.append(raw)
        except Exception as exc:
            raise StorageValidationError(
                store_name=self.store_name,
                path=self.path,
                operation=operation,
                reason=str(exc) or "registry item validation failed",
                original_error=exc,
            ) from exc
        return dumped

    def _payload(
        self,
        *,
        revision: int,
        updated_at: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "revision": revision,
            "updated_at": updated_at,
            "content_hash": _content_hash(
                schema_version=2,
                revision=revision,
                updated_at=updated_at,
                items=items,
            ),
            "items": items,
        }

    def _write_payload(self, payload: dict[str, Any]) -> None:
        content = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            + b"\n"
        )
        self.atomic_store.write_bytes_assume_locked(
            self.path,
            content,
            validator=lambda raw: self._validate_encoded_payload(raw, payload),
        )

    def _validate_encoded_payload(self, raw: bytes, expected: dict[str, Any]) -> bool:
        decoded = cast(dict[str, Any], json.loads(raw))
        return (
            decoded == expected and self._parse_v2(decoded).content_hash == expected["content_hash"]
        )

    def _schema_error(self, reason: str) -> StorageSchemaMismatchError:
        return StorageSchemaMismatchError(
            store_name=self.store_name,
            path=self.path,
            operation="load",
            reason=reason,
            recoverable=False,
            suggested_repair="migrate or restore a registry matching the supported schema",
        )


def _content_hash(
    *,
    schema_version: int,
    revision: int,
    updated_at: str,
    items: list[Any],
) -> str:
    canonical = json.dumps(
        {
            "schema_version": schema_version,
            "revision": revision,
            "updated_at": updated_at,
            "items": items,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
