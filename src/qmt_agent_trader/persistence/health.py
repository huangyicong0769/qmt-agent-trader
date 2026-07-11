"""Single secret-safe storage health payload mapper."""

from __future__ import annotations

from typing import Any, Literal

from qmt_agent_trader.agent.audit import scrub_sensitive
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageCorruptError,
    StorageError,
    StorageLockTimeoutError,
)

StorageStatus = Literal["ok", "degraded", "corrupt", "locked", "conflict"]


def storage_health_payload(
    *,
    component: str,
    status: StorageStatus,
    reason: str = "",
    warnings: list[str] | None = None,
    repair_action: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "storage_status": status,
        "storage_component": component,
        "storage_reason": reason,
        "storage_warnings": warnings or [],
        "storage_repair_action": repair_action,
    }
    scrubbed = scrub_sensitive(payload)
    if not isinstance(scrubbed, dict):  # pragma: no cover
        raise TypeError("storage health payload must remain an object")
    return scrubbed


def storage_error_health_payload(exc: StorageError) -> dict[str, Any]:
    if isinstance(exc, StorageCorruptError):
        status: StorageStatus = "corrupt"
    elif isinstance(exc, StorageLockTimeoutError):
        status = "locked"
    elif isinstance(exc, StorageConflictError):
        status = "conflict"
    else:
        status = "degraded"
    return {
        "status": "STORAGE_ERROR",
        **storage_health_payload(
            component=exc.store_name,
            status=status,
            reason=exc.reason,
            repair_action=exc.suggested_repair,
        ),
    }
