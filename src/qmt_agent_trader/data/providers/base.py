"""Provider contracts for registry-driven data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderCapability:
    source: str
    endpoints: list[dict[str, Any]]


@dataclass(frozen=True)
class FetchItem:
    api_name: str
    symbols: list[str] = field(default_factory=list)
    fields: list[str] | None = None
    trade_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchPlan:
    status: str
    source: str
    items: list[dict[str, Any]] = field(default_factory=list)
    estimated_request_count: int = 0
    reason: str | None = None
    message: str | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "source": self.source,
            "estimated_request_count": self.estimated_request_count,
            "items": self.items,
            "errors": self.errors,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class FetchResult:
    status: str
    source: str
    writes: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "writes": self.writes,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class DataSourceProvider(Protocol):
    source_name: str

    def list_capabilities(
        self,
        *,
        category: str | None = None,
        asset_type: str | None = None,
    ) -> ProviderCapability:
        ...

    def plan_fetch(
        self,
        items: list[FetchItem],
        *,
        requested_by_llm: bool = False,
        storage_mode: str = "persistent",
    ) -> FetchPlan:
        ...

    def run_fetch(
        self,
        plan: FetchPlan,
        *,
        execute_plan: bool = False,
        dry_run: bool = False,
    ) -> FetchResult:
        ...
