"""Filesystem registry for saved universe specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.repositories.versioned_record import (
    RecordDiagnostic,
    VersionedRecordRepository,
)
from qmt_agent_trader.universe.models import UniverseSpec


class UniverseStoredRecord(BaseModel):
    schema_version: int = 2
    revision: int = 0
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
        )
        self.last_diagnostics: list[RecordDiagnostic] = []

    @classmethod
    def for_lake(cls, lake: DataLake) -> UniverseRegistry:
        return cls(lake.root.parent / "universes" / "registry")

    def save(self, spec: UniverseSpec) -> Path:
        if spec.source == "agent_generated":
            spec.research_only = True
            spec.live_trading_allowed = False
            spec.approval_required = True
        path = self.path_for(spec.universe_id)
        record = UniverseStoredRecord(spec=spec)
        try:
            self.repository.load(spec.universe_id)
        except FileNotFoundError:
            self.repository.create(spec.universe_id, record)
        else:
            self.repository.mutate(spec.universe_id, lambda _old: record)
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
    if raw:
        return Path(str(raw))
    if lake is not None:
        return UniverseRegistry.for_lake(lake).root
    return Path("data/universes/registry")
