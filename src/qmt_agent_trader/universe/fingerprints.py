"""Stable fingerprints for universe specs and resolved symbols."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qmt_agent_trader.universe.models import UniverseSpec


def fingerprint_spec(spec: UniverseSpec) -> str:
    payload = spec.model_dump(mode="json")
    return _sha(payload)


def fingerprint_symbols(
    spec: UniverseSpec,
    *,
    mode: str,
    symbols: list[str] | None = None,
    rolling_symbols: dict[str, list[str]] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "spec": spec.model_dump(mode="json"),
        "mode": mode,
        "symbols": symbols or [],
        "rolling_symbols": rolling_symbols or {},
    }
    return _sha(payload)


def _sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
