"""Disposable content-addressed JSON cache with bounded lifetime."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore, FaultHook
from qmt_agent_trader.persistence.errors import (
    StorageError,
    StorageLockTimeoutError,
    StorageValidationError,
)

WarningSink = Callable[[dict[str, object]], None]
_NAMESPACE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_KEY = re.compile(r"^[0-9a-f]{64}$")


class ContentAddressedCache:
    def __init__(
        self,
        root: Path,
        atomic_store: AtomicFileStore,
        *,
        ttl: timedelta = timedelta(days=1),
        clock: Callable[[], datetime] | None = None,
        warning_sink: WarningSink | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.atomic_store = atomic_store
        self.ttl = ttl
        self.clock = clock or (lambda: datetime.now(UTC))
        self.warning_sink = warning_sink or self._log_warning
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "expired": 0,
            "corrupt_invalidations": 0,
            "invalidations": 0,
            "invalidation_failures": 0,
            "read_failures": 0,
            "write_failures": 0,
        }

    @staticmethod
    def key_for(value: Any) -> str:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def path_for(self, namespace: str, key: str) -> Path:
        if not _NAMESPACE.fullmatch(namespace) or not _KEY.fullmatch(key):
            raise StorageValidationError(
                store_name="cache",
                path=self.root,
                operation="path_for",
                reason="namespace or key is not canonical",
            )
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self.path_for(namespace, key)
        with self.atomic_store.lock_manager.resource_lock(path):
            if not path.exists():
                self.metrics["misses"] += 1
                return None
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                if not self._valid_envelope(envelope, key):
                    raise ValueError("invalid cache envelope")
                if datetime.fromisoformat(envelope["expires_at"]) <= self.clock():
                    self.metrics["expired"] += 1
                    self._invalidate_unlocked(path)
                    return None
                self.metrics["hits"] += 1
                return dict(envelope["value"])
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
                self.metrics["corrupt_invalidations"] += 1
                self._invalidate_unlocked(path)
                self._warn(
                    {
                        "reason": "CACHE_CORRUPT_INVALIDATED",
                        "path": str(path),
                        "error": type(exc).__name__,
                    }
                )
                return None

    def put(
        self,
        namespace: str,
        key: str,
        value: dict[str, Any],
        *,
        fault_hook: FaultHook | None = None,
        raise_on_error: bool = False,
    ) -> None:
        now = self.clock()
        envelope = {
            "schema_version": 1,
            "key": key,
            "created_at": now.isoformat(),
            "expires_at": (now + self.ttl).isoformat(),
            "value": value,
        }
        path = self.path_for(namespace, key)
        try:
            with self.atomic_store.lock_manager.resource_lock(path):
                self.atomic_store.write_json(
                    path,
                    envelope,
                    validator=lambda item: self._valid_envelope(item, key),
                    fault_hook=fault_hook,
                )
        except StorageLockTimeoutError:
            raise
        except StorageError as exc:
            self.metrics["write_failures"] += 1
            self._warn(
                {
                    "reason": "CACHE_WRITE_FAILED",
                    "path": str(path),
                    "error": type(exc).__name__,
                }
            )
            if raise_on_error:
                raise

    def invalidate(
        self,
        namespace: str,
        key: str,
        *,
        expected_value: dict[str, Any] | None = None,
        reason: str = "CACHE_INVALIDATED",
    ) -> bool:
        path = self.path_for(namespace, key)
        try:
            with self.atomic_store.lock_manager.resource_lock(path):
                if not path.exists():
                    return False
                if expected_value is not None:
                    envelope = json.loads(path.read_text(encoding="utf-8"))
                    if not self._valid_envelope(envelope, key):
                        return False
                    if envelope["value"] != expected_value:
                        return False
                removed = self._invalidate_unlocked(path)
                if removed:
                    self.metrics["invalidations"] += 1
                    self._warn({"reason": reason, "path": str(path)})
                return removed
        except Exception as exc:
            self.metrics["invalidation_failures"] += 1
            self._warn(
                {
                    "reason": "CACHE_INVALIDATION_FAILED",
                    "path": str(path),
                    "error": type(exc).__name__,
                }
            )
            return False

    def _invalidate_unlocked(self, path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            self.metrics["invalidation_failures"] += 1
            self._warn(
                {
                    "reason": "CACHE_INVALIDATION_FAILED",
                    "path": str(path),
                    "error": type(exc).__name__,
                }
            )
            return False

    def _warn(self, payload: dict[str, object]) -> None:
        try:
            self.warning_sink(payload)
        except Exception:
            logging.getLogger(__name__).exception("cache warning sink failed")

    @staticmethod
    def _valid_envelope(value: Any, key: str) -> bool:
        return (
            isinstance(value, dict)
            and value.get("schema_version") == 1
            and value.get("key") == key
            and isinstance(value.get("value"), dict)
            and isinstance(value.get("expires_at"), str)
        )

    @staticmethod
    def _log_warning(payload: dict[str, object]) -> None:
        logging.getLogger(__name__).warning("cache invalidated", extra={"cache": payload})
