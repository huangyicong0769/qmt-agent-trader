"""Filesystem registry for saved universe specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseSpec


class UniverseRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def for_lake(cls, lake: DataLake) -> UniverseRegistry:
        return cls(lake.root.parent / "universes" / "registry")

    def save(self, spec: UniverseSpec) -> Path:
        if spec.source == "agent_generated":
            spec.research_only = True
            spec.live_trading_allowed = False
            spec.approval_required = True
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(spec.universe_id)
        path.write_text(
            json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load(self, universe_id: str) -> UniverseSpec | None:
        path = self.path_for(universe_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return UniverseSpec.model_validate(payload)

    def list(
        self,
        *,
        source: str | None = None,
        query: str | None = None,
        asset_type: str | None = None,
        mode: str | None = None,
    ) -> list[UniverseSpec]:
        specs: list[UniverseSpec] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                spec = UniverseSpec.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
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
        safe = universe_id.replace("/", "_")
        return self.root / f"{safe}.json"


def registry_root_from_payload(payload: dict[str, Any], lake: DataLake | None) -> Path:
    raw = payload.get("registry_root")
    if raw:
        return Path(str(raw))
    if lake is not None:
        return UniverseRegistry.for_lake(lake).root
    return Path("data/universes/registry")
