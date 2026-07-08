"""Helpers for structured data-tool outcomes."""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    ExecutionStatus,
    RecommendationStatus,
)


def data_tool_ok(
    *,
    status: str = "OK",
    coverage_status: str = "OK",
    message: str | None = None,
    recommendation_status: str = RecommendationStatus.RESEARCH_ONLY.value,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.OK.value,
        evidence_status=EvidenceStatus.VALID.value,
        recommendation_status=recommendation_status,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def data_tool_partial(
    *,
    status: str = "PARTIAL_COVERAGE",
    coverage_status: str = "PARTIAL_COVERAGE",
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.PARTIAL.value,
        evidence_status=EvidenceStatus.INCOMPLETE.value,
        recommendation_status=RecommendationStatus.UNKNOWN.value,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def data_tool_no_data(
    *,
    status: str = "NO_DATA",
    coverage_status: str = "NO_DATA",
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.NO_DATA.value,
        evidence_status=EvidenceStatus.INCOMPLETE.value,
        recommendation_status=RecommendationStatus.BLOCKED.value,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def data_tool_invalid_request(
    *,
    status: str = "INVALID_REQUEST",
    coverage_status: str = "INVALID_REQUEST",
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.INVALID_REQUEST.value,
        evidence_status=EvidenceStatus.INVALID.value,
        recommendation_status=RecommendationStatus.BLOCKED.value,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def data_tool_blocked(
    *,
    status: str = "BLOCKED",
    coverage_status: str = "BLOCKED",
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.BLOCKED.value,
        evidence_status=EvidenceStatus.BLOCKED.value,
        recommendation_status=RecommendationStatus.BLOCKED.value,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def data_tool_not_configured(
    *,
    status: str = "NOT_CONFIGURED",
    coverage_status: str = "BLOCKED",
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.NOT_CONFIGURED.value,
        evidence_status=EvidenceStatus.BLOCKED.value,
        recommendation_status=RecommendationStatus.BLOCKED.value,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def data_tool_failed(
    *,
    status: str = "FAILED",
    coverage_status: str = "INVALID_REQUEST",
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.FAILED.value,
        evidence_status=EvidenceStatus.INVALID.value,
        recommendation_status=RecommendationStatus.BLOCKED.value,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def data_tool_weak(
    *,
    status: str,
    coverage_status: str = "NOT_VERIFIED",
    message: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _outcome(
        status=status,
        domain_status=DomainStatus.WARN.value,
        evidence_status=EvidenceStatus.WEAK.value,
        recommendation_status=RecommendationStatus.RESEARCH_ONLY.value,
        coverage_status=coverage_status,
        message=message,
        **extra,
    )


def _outcome(
    *,
    status: str,
    domain_status: str,
    evidence_status: str,
    recommendation_status: str,
    coverage_status: str,
    message: str | None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "execution_status": ExecutionStatus.OK.value,
        "domain_status": domain_status,
        "evidence_status": evidence_status,
        "recommendation_status": recommendation_status,
        "coverage_status": coverage_status,
        "raw_status": status,
        "message": message,
        "reason": extra.pop("reason", None),
        "warnings": list(extra.pop("warnings", []) or []),
        "blockers": list(extra.pop("blockers", []) or []),
        "next_repair_tool": extra.pop("next_repair_tool", None),
        "suggested_repair": extra.pop("suggested_repair", None),
        "repair_action": extra.pop("repair_action", None),
        "verification_action": extra.pop("verification_action", None),
    }
    payload.update(extra)
    return payload
