"""Tushare endpoint registry loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EndpointSpec:
    api_name: str
    dataset_id: str
    category: str
    asset_types: tuple[str, ...]
    implemented: bool
    doc_url: str | None
    doc_status: str
    params: dict[str, dict[str, Any]]
    fields: tuple[str, ...]
    default_fields: tuple[str, ...]
    key_columns: tuple[str, ...]
    symbol_param: str | None
    symbol_column: str | None
    date_params: dict[str, dict[str, Any]]
    supports_symbol_range: bool
    supports_marketwide_by_date: bool
    pagination: dict[str, Any]
    pit: dict[str, Any]
    wide_table_targets: tuple[str, ...]
    raw_dataset_name: str
    raw_view_name: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EndpointSpec:
        return cls(
            api_name=str(data["api_name"]),
            dataset_id=str(data["dataset_id"]),
            category=str(data["category"]),
            asset_types=tuple(str(item) for item in data.get("asset_types", [])),
            implemented=bool(data.get("implemented", False)),
            doc_url=data.get("doc_url"),
            doc_status=str(data.get("doc_status", "DOC_UNAVAILABLE")),
            params=dict(data.get("params", {})),
            fields=tuple(str(item) for item in data.get("fields", [])),
            default_fields=tuple(str(item) for item in data.get("default_fields", [])),
            key_columns=tuple(str(item) for item in data.get("key_columns", [])),
            symbol_param=data.get("symbol_param"),
            symbol_column=data.get("symbol_column"),
            date_params=dict(data.get("date_params", {})),
            supports_symbol_range=bool(data.get("supports_symbol_range", False)),
            supports_marketwide_by_date=bool(data.get("supports_marketwide_by_date", False)),
            pagination=dict(data.get("pagination", {"type": "none"})),
            pit=dict(data.get("pit", {})),
            wide_table_targets=tuple(str(item) for item in data.get("wide_table_targets", [])),
            raw_dataset_name=str(data["raw_dataset_name"]),
            raw_view_name=str(data["raw_view_name"]),
        )

    def as_capability(self) -> dict[str, Any]:
        return {
            "api_name": self.api_name,
            "dataset_id": self.dataset_id,
            "category": self.category,
            "asset_types": list(self.asset_types),
            "implemented": self.implemented,
            "doc_url": self.doc_url,
            "doc_status": self.doc_status,
            "params": self.params,
            "fields": list(self.fields),
            "default_fields": list(self.default_fields),
            "key_columns": list(self.key_columns),
            "symbol_param": self.symbol_param,
            "symbol_column": self.symbol_column,
            "date_params": self.date_params,
            "supports_symbol_range": self.supports_symbol_range,
            "supports_marketwide_by_date": self.supports_marketwide_by_date,
            "pagination": self.pagination,
            "pit": self.pit,
            "wide_table_targets": list(self.wide_table_targets),
            "raw_dataset_name": self.raw_dataset_name,
            "raw_view_name": self.raw_view_name,
        }


class TushareEndpointRegistry:
    def __init__(self, specs: list[EndpointSpec]) -> None:
        self._specs = {spec.api_name: spec for spec in specs}

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> TushareEndpointRegistry:
        resolved = path or Path(__file__).with_name("endpoints.yml")
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        endpoints = data.get("endpoints", [])
        if not isinstance(endpoints, list):
            raise ValueError("endpoints.yml must contain an endpoints list")
        specs = [EndpointSpec.from_mapping(item) for item in endpoints]
        _validate_specs(specs)
        return cls(specs)

    def get(self, api_name: str) -> EndpointSpec | None:
        return self._specs.get(api_name)

    def require(self, api_name: str) -> EndpointSpec:
        spec = self.get(api_name)
        if spec is None:
            raise KeyError(api_name)
        return spec

    def list_endpoints(
        self,
        *,
        category: str | None = None,
        asset_type: str | None = None,
    ) -> list[EndpointSpec]:
        specs = sorted(self._specs.values(), key=lambda item: item.api_name)
        if category:
            specs = [spec for spec in specs if spec.category == category]
        if asset_type:
            specs = [spec for spec in specs if asset_type in spec.asset_types]
        return specs

    def as_capabilities(
        self,
        *,
        category: str | None = None,
        asset_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            spec.as_capability()
            for spec in self.list_endpoints(category=category, asset_type=asset_type)
        ]


@lru_cache(maxsize=1)
def default_tushare_registry() -> TushareEndpointRegistry:
    return TushareEndpointRegistry.from_yaml()


def _validate_specs(specs: list[EndpointSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.api_name in seen:
            raise ValueError(f"duplicate Tushare endpoint: {spec.api_name}")
        seen.add(spec.api_name)
        if spec.implemented and not spec.fields:
            raise ValueError(f"implemented endpoint has no fields: {spec.api_name}")
        if spec.implemented and not spec.key_columns:
            raise ValueError(f"implemented endpoint has no key columns: {spec.api_name}")
        missing_defaults = set(spec.default_fields).difference(spec.fields)
        if missing_defaults:
            raise ValueError(
                f"default fields not present for {spec.api_name}: {sorted(missing_defaults)}"
            )
        missing_keys = set(spec.key_columns).difference(spec.fields)
        if missing_keys:
            raise ValueError(
                f"key columns not present for {spec.api_name}: {sorted(missing_keys)}"
            )
