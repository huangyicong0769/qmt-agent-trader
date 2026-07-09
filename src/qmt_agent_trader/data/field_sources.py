"""Field-source discovery from Tushare endpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from qmt_agent_trader.data.frequency import Frequency, is_lower_frequency
from qmt_agent_trader.data.macro import get_macro_dataset
from qmt_agent_trader.data.providers.tushare.registry import (
    EndpointSpec,
    TushareEndpointRegistry,
)


class FillPolicy(StrEnum):
    EXACT = "exact"
    ASOF_SNAPSHOT = "asof_snapshot"
    EVENT_TO_STATE = "event_to_state"
    NO_FILL = "no_fill"


@dataclass(frozen=True)
class FieldSourceSpec:
    field: str
    api_name: str
    raw_dataset_name: str
    entity_column: str | None
    canonical_entity_column: str
    source_time_column: str | None
    visible_time_column: str | None
    fallback_visible_time_column: str | None
    key_columns: tuple[str, ...]
    frequency: Frequency
    fill_policy: FillPolicy
    pit_safe: bool

    def as_metadata(self) -> dict[str, Any]:
        return {
            "api_name": self.api_name,
            "raw_dataset_name": self.raw_dataset_name,
            "frequency": self.frequency.value,
            "fill_policy": self.fill_policy.value,
            "pit_safe": self.pit_safe,
        }


class FieldSourceIndex:
    def __init__(self, sources: list[FieldSourceSpec]) -> None:
        by_field: dict[str, list[FieldSourceSpec]] = {}
        for source in sources:
            by_field.setdefault(source.field, []).append(source)
        self._by_field = {
            field: sorted(items, key=lambda item: item.api_name)
            for field, items in by_field.items()
        }

    @classmethod
    def from_tushare_registry(cls, registry: TushareEndpointRegistry) -> FieldSourceIndex:
        sources: list[FieldSourceSpec] = []
        for endpoint in registry.list_endpoints():
            if not endpoint.implemented:
                continue
            for field in endpoint.fields:
                sources.append(_field_source_from_endpoint(endpoint, field))
        return cls(sources)

    def sources_for_field(self, field: str) -> list[FieldSourceSpec]:
        return list(self._by_field.get(field, ()))

    def best_source_for_field(
        self,
        field: str,
        *,
        target_frequency: Frequency,
        preferred_api: str | None = None,
    ) -> FieldSourceSpec | None:
        candidates = self.sources_for_field(field)
        if preferred_api is not None:
            candidates = [source for source in candidates if source.api_name == preferred_api]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None

        exact = [
            source
            for source in candidates
            if source.frequency == target_frequency and source.fill_policy is FillPolicy.EXACT
        ]
        if len(exact) == 1:
            return exact[0]

        asof = [
            source
            for source in candidates
            if source.fill_policy is FillPolicy.ASOF_SNAPSHOT and source.pit_safe
        ]
        if len(asof) == 1:
            return asof[0]
        return None


def fetch_columns_for_source(source: FieldSourceSpec, fields: list[str]) -> list[str]:
    """Return raw columns needed to fetch ``fields`` from ``source`` safely."""

    identity: list[str] = []
    if source.entity_column is not None:
        identity.append(source.entity_column)
    if source.visible_time_column and source.visible_time_column != "visible_date":
        identity.append(source.visible_time_column)
    if source.fallback_visible_time_column:
        identity.append(source.fallback_visible_time_column)
    if source.source_time_column is not None:
        identity.append(source.source_time_column)
    identity.extend(source.key_columns)
    return list(dict.fromkeys([*identity, *fields]))


def _field_source_from_endpoint(endpoint: EndpointSpec, field: str) -> FieldSourceSpec:
    frequency = _infer_frequency(endpoint)
    visible_column = _visible_time_column(endpoint)
    fallback_column = _fallback_visible_time_column(endpoint)
    pit_safe = _pit_safe(endpoint, visible_column, fallback_column)
    fill_policy = _infer_fill_policy(
        frequency,
        pit_safe=pit_safe,
        visible_time_column=visible_column,
    )
    return FieldSourceSpec(
        field=field,
        api_name=endpoint.api_name,
        raw_dataset_name=endpoint.raw_dataset_name,
        entity_column=endpoint.symbol_column,
        canonical_entity_column="symbol",
        source_time_column=_source_time_column(endpoint),
        visible_time_column=visible_column,
        fallback_visible_time_column=fallback_column,
        key_columns=endpoint.key_columns,
        frequency=frequency,
        fill_policy=fill_policy,
        pit_safe=pit_safe,
    )


def _infer_frequency(endpoint: EndpointSpec) -> Frequency:
    macro = get_macro_dataset(endpoint.api_name)
    if macro is not None:
        return _frequency_from_string(macro.frequency)
    if endpoint.category in {"corporate_action"}:
        return Frequency.EVENT
    if endpoint.category in {"security"} and "trade_date" not in endpoint.fields:
        return Frequency.EVENT

    visible_column = endpoint.pit.get("visible_date_column")
    if "trade_date" in endpoint.fields and visible_column == "trade_date":
        return Frequency.DAILY
    if "date" in endpoint.fields and visible_column == "date":
        return Frequency.DAILY
    if _has_any(endpoint, {"month", "start_m", "end_m"}):
        return Frequency.MONTHLY
    if _has_any(endpoint, {"quarter", "start_q", "end_q"}):
        return Frequency.QUARTERLY
    if "end_date" in endpoint.fields and _has_any(endpoint, {"ann_date", "f_ann_date"}):
        return Frequency.QUARTERLY
    return Frequency.UNKNOWN


def _infer_fill_policy(
    frequency: Frequency,
    *,
    pit_safe: bool,
    visible_time_column: str | None,
) -> FillPolicy:
    if frequency is Frequency.EVENT:
        return FillPolicy.NO_FILL
    if frequency is Frequency.UNKNOWN:
        return FillPolicy.NO_FILL
    if frequency is Frequency.DAILY:
        return FillPolicy.EXACT
    if (
        is_lower_frequency(frequency, Frequency.DAILY)
        and pit_safe
        and visible_time_column is not None
    ):
        return FillPolicy.ASOF_SNAPSHOT
    return FillPolicy.NO_FILL


def _visible_time_column(endpoint: EndpointSpec) -> str | None:
    if get_macro_dataset(endpoint.api_name) is not None:
        return "visible_date"
    value = endpoint.pit.get("visible_date_column")
    return str(value) if value else None


def _fallback_visible_time_column(endpoint: EndpointSpec) -> str | None:
    value = endpoint.pit.get("fallback_visible_date_column")
    return str(value) if value else None


def _source_time_column(endpoint: EndpointSpec) -> str | None:
    macro = get_macro_dataset(endpoint.api_name)
    if macro is not None:
        return macro.date_column
    for column in ("trade_date", "end_date", "period", "month", "quarter", "date"):
        if column in endpoint.fields:
            return column
    visible = endpoint.pit.get("visible_date_column")
    return str(visible) if visible else None


def _pit_safe(
    endpoint: EndpointSpec,
    visible_time_column: str | None,
    fallback_visible_time_column: str | None,
) -> bool:
    macro = get_macro_dataset(endpoint.api_name)
    if macro is not None:
        return bool(macro.date_column and visible_time_column)
    return bool(endpoint.pit.get("safe")) and (
        visible_time_column is not None or fallback_visible_time_column is not None
    )


def _has_any(endpoint: EndpointSpec, names: set[str]) -> bool:
    return bool(names.intersection(endpoint.fields)) or bool(names.intersection(endpoint.params))


def _frequency_from_string(value: str) -> Frequency:
    try:
        return Frequency(value)
    except ValueError:
        return Frequency.UNKNOWN
