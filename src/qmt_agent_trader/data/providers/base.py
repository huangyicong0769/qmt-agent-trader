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
    execution_status: str = "OK"
    domain_status: str = "UNKNOWN"
    evidence_status: str = "UNKNOWN"
    recommendation_status: str = "UNKNOWN"
    coverage_status: str = "UNKNOWN"
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_repair_tool: str | None = None
    suggested_repair: dict[str, Any] | None = None
    repair_action: dict[str, Any] | None = None
    verification_action: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        status_fields = _structured_fields_for_plan(
            status=self.status,
            execution_status=self.execution_status,
            domain_status=self.domain_status,
            evidence_status=self.evidence_status,
            recommendation_status=self.recommendation_status,
            coverage_status=self.coverage_status,
        )
        payload: dict[str, Any] = {
            "status": self.status,
            **status_fields,
            "source": self.source,
            "estimated_request_count": self.estimated_request_count,
            "items": self.items,
            "errors": self.errors,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "next_repair_tool": self.next_repair_tool,
            "suggested_repair": self.suggested_repair,
            "repair_action": self.repair_action,
            "verification_action": self.verification_action,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.message:
            payload["message"] = self.message
        payload.update(self.metadata)
        return payload


@dataclass(frozen=True)
class FetchResult:
    status: str
    source: str
    writes: list[dict[str, Any]] = field(default_factory=list)
    dataset_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "OK"
    domain_status: str = "UNKNOWN"
    evidence_status: str = "UNKNOWN"
    recommendation_status: str = "UNKNOWN"
    coverage_status: str = "UNKNOWN"
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_repair_tool: str | None = None
    suggested_repair: dict[str, Any] | None = None
    repair_action: dict[str, Any] | None = None
    verification_action: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "execution_status": self.execution_status,
            "domain_status": self.domain_status,
            "evidence_status": self.evidence_status,
            "recommendation_status": self.recommendation_status,
            "coverage_status": self.coverage_status,
            "raw_status": self.status,
            "source": self.source,
            "writes": self.writes,
            "dataset_results": self.dataset_results,
            "errors": self.errors,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "next_repair_tool": self.next_repair_tool,
            "suggested_repair": self.suggested_repair,
            "repair_action": self.repair_action,
            "verification_action": self.verification_action,
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


def _structured_fields_for_plan(
    *,
    status: str,
    execution_status: str,
    domain_status: str,
    evidence_status: str,
    recommendation_status: str,
    coverage_status: str,
) -> dict[str, str]:
    if (
        domain_status != "UNKNOWN"
        or evidence_status != "UNKNOWN"
        or recommendation_status != "UNKNOWN"
        or coverage_status != "UNKNOWN"
    ):
        return {
            "execution_status": execution_status,
            "domain_status": domain_status,
            "evidence_status": evidence_status,
            "recommendation_status": recommendation_status,
            "coverage_status": coverage_status,
            "raw_status": status,
        }

    normalized = status.upper()
    if normalized == "PLANNED":
        domain, evidence, recommendation, coverage = (
            "OK",
            "WEAK",
            "RESEARCH_ONLY",
            "NOT_VERIFIED",
        )
    elif normalized in {"BLOCKED", "NOT_IMPLEMENTED"}:
        domain, evidence, recommendation, coverage = (
            "BLOCKED",
            "BLOCKED",
            "BLOCKED",
            "BLOCKED",
        )
    elif normalized == "INVALID_REQUEST":
        domain, evidence, recommendation, coverage = (
            "INVALID_REQUEST",
            "INVALID",
            "BLOCKED",
            "INVALID_REQUEST",
        )
    else:
        domain, evidence, recommendation, coverage = (
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
        )
    return {
        "execution_status": execution_status,
        "domain_status": domain,
        "evidence_status": evidence,
        "recommendation_status": recommendation,
        "coverage_status": coverage,
        "raw_status": status,
    }
