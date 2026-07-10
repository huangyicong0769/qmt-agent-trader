"""Filesystem registry for saved universe specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from qmt_agent_trader.data.storage import DataLake
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

    @classmethod
    def for_lake(cls, lake: DataLake) -> UniverseRegistry:
        data_root = lake.root.parent.resolve()
        return cls(
            data_root / "registries" / "universes",
            locks_root=data_root / "locks",
            quarantine_root=data_root / "quarantine" / "universes",
        )

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

    def list(
        self,
        *,
        source: str | None = None,
        query: str | None = None,
        asset_type: str | None = None,
        mode: str | None = None,
    ) -> list[UniverseSpec]:
        records, self.last_diagnostics = self.repository.list_with_diagnostics()
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
        canonical = UniverseRegistry.for_lake(lake).root
        if raw and Path(str(raw)).expanduser().resolve() != canonical.resolve():
            raise ValueError("registry_root override must equal the canonical registry root")
        return canonical
    if raw:
        raise ValueError("registry_root override requires a configured data lake")
    raise ValueError("universe registry requires a configured data lake")
