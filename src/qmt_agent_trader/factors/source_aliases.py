"""Canonical factor fields backed by differently named raw fields."""

from __future__ import annotations

from dataclasses import dataclass

from qmt_agent_trader.data.field_sources import FieldSourceIndex, FieldSourceSpec
from qmt_agent_trader.data.frequency import Frequency


@dataclass(frozen=True)
class CanonicalFieldSource:
    canonical_field: str
    source_field: str
    source: FieldSourceSpec


_CANONICAL_ALIASES: dict[str, tuple[str, str]] = {
    "turnover": ("daily_basic", "turnover_rate"),
}


def resolve_canonical_field_source(
    index: FieldSourceIndex,
    field: str,
    *,
    target_frequency: Frequency,
) -> CanonicalFieldSource | None:
    alias = _CANONICAL_ALIASES.get(field)
    if alias is None:
        source = index.best_source_for_field(
            field,
            target_frequency=target_frequency,
        )
        return None if source is None else CanonicalFieldSource(field, field, source)
    api_name, source_field = alias
    source = index.best_source_for_field(
        source_field,
        target_frequency=target_frequency,
        preferred_api=api_name,
    )
    return (
        None
        if source is None
        else CanonicalFieldSource(field, source_field, source)
    )
