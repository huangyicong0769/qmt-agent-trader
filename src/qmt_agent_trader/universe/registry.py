"""Filesystem registry for saved universe specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.errors import (
    StorageCorruptError,
    StorageError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.repositories.versioned_record import (
    RecordDiagnostic,
    VersionedRecordRepository,
)
from qmt_agent_trader.universe.models import UniverseSpec


class UniverseStoredRecord(BaseModel):
    schema_version: Literal[2] = 2
    revision: int = Field(default=0, ge=0)
    updated_at: str = ""
    spec: UniverseSpec

    @model_validator(mode="before")
    @classmethod
    def wrap_legacy(cls, value: Any) -> Any:
        if isinstance(value, dict) and "spec" not in value and "universe_id" in value:
            return {"spec": value}
        return value


class UniverseRegistry:
    def __init__(
        self, root: Path, *, locks_root: Path | None = None, quarantine_root: Path | None = None
    ) -> None:
        self.root = root.expanduser().resolve()
        self.repository = VersionedRecordRepository(
            self.root,
            UniverseStoredRecord,
            store_name="universes",
            locks_root=locks_root,
            quarantine_root=quarantine_root,
            identity=lambda record: record.spec.universe_id,
        )
        self.last_diagnostics: list[RecordDiagnostic] = []
        self._previous_root_diagnostics: list[RecordDiagnostic] = []

    @classmethod
    def for_lake(cls, lake: DataLake) -> UniverseRegistry:
        data_root = lake.root.parent.resolve()
        registry = cls(
            data_root / "registries" / "universes",
            locks_root=data_root / "locks",
            quarantine_root=data_root / "quarantine" / "universes",
        )
        registry._migrate_previous_root(data_root / "universes" / "registry")
        return registry

    def _migrate_previous_root(self, previous_root: Path) -> None:
        if not previous_root.exists() or previous_root.resolve() == self.root:
            return
        migration_resource = self.root.parent / ".universe-root-migration"
        with self.repository.lock_manager.resource_lock(migration_resource):
            legacy_repository = VersionedRecordRepository(
                previous_root,
                UniverseStoredRecord,
                store_name="universes_previous_root",
                locks_root=self.repository.lock_manager.locks_root,
                quarantine_root=self.repository.quarantine_root / "legacy-root",
                identity=lambda record: record.spec.universe_id,
            )
            records: list[UniverseStoredRecord] = []
            diagnostics: list[RecordDiagnostic] = []
            for path in sorted(previous_root.glob("*.json")):
                try:
                    raw = path.read_bytes()
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("universe record root must be an object")
                    if "schema_version" in payload:
                        record = legacy_repository._load_locked(path, migrate=False)
                    else:
                        record = UniverseStoredRecord.model_validate(payload)
                    if record.spec.universe_id != path.stem:
                        raise ValueError("universe id does not match previous storage key")
                    records.append(record)
                except StorageError as exc:
                    diagnostics.append(RecordDiagnostic(path, exc))
                except (
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    error_type = (
                        StorageCorruptError
                        if isinstance(exc, (OSError, UnicodeDecodeError, json.JSONDecodeError))
                        else StorageValidationError
                    )
                    diagnostics.append(RecordDiagnostic(path, error_type(
                        store_name="universes_previous_root", path=path,
                        operation="discover", reason="previous universe record is invalid",
                        original_error=exc)))
            self._previous_root_diagnostics = diagnostics
            for record in records:
                if self.load(record.spec.universe_id) is None:
                    self.repository.create(record.spec.universe_id, record)

    def save(self, spec: UniverseSpec, *, expected_revision: int | None = None) -> Path:
        if spec.source == "agent_generated":
            spec.research_only = True
            spec.live_trading_allowed = False
            spec.approval_required = True
        path = self.path_for(spec.universe_id)
        record = UniverseStoredRecord(spec=spec)
        self.repository.upsert(
            spec.universe_id,
            lambda _old: record,
            expected_revision=expected_revision,
        )
        return path

    def load(self, universe_id: str) -> UniverseSpec | None:
        try:
            return self.repository.load(universe_id).spec
        except FileNotFoundError:
            return None

    def load_record(self, universe_id: str) -> UniverseStoredRecord | None:
        try:
            return self.repository.load(universe_id)
        except FileNotFoundError:
            return None

    def list(
        self,
        *,
        source: str | None = None,
        query: str | None = None,
        asset_type: str | None = None,
        mode: str | None = None,
    ) -> list[UniverseSpec]:
        records, canonical_diagnostics = self.repository.list_with_diagnostics()
        self.last_diagnostics = [
            *self._previous_root_diagnostics,
            *canonical_diagnostics,
        ]
        specs: list[UniverseSpec] = []
        for record in records:
            spec = record.spec
            if source and spec.source != source:
                continue
            if asset_type and asset_type not in spec.asset_types:
                continue
            if mode and spec.mode != mode:
                continue
            if query:
                haystack = f"{spec.universe_id} {spec.name} {spec.description}".lower()
                if query.lower() not in haystack:
                    continue
            specs.append(spec)
        return specs

    def path_for(self, universe_id: str) -> Path:
        return self.repository.path_for(universe_id)


def registry_root_from_payload(payload: dict[str, Any], lake: DataLake | None) -> Path:
    raw = payload.get("registry_root")
    if lake is not None:
        canonical = lake.root.parent.resolve() / "registries" / "universes"
        if raw and Path(str(raw)).expanduser().resolve() != canonical.resolve():
            raise ValueError("registry_root override must equal the canonical registry root")
        return canonical
    if raw:
        raise ValueError("registry_root override requires a configured data lake")
    raise ValueError("universe registry requires a configured data lake")
