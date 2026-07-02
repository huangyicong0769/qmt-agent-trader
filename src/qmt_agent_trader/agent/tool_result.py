"""Structured tool-result status helpers.

The helpers in this module make evidence state observable. They deliberately
avoid optimistic success inference for legacy payloads: a Python function may
return successfully while the research evidence remains unknown, blocked, or
invalid.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ExecutionStatus(StrEnum):
    STARTED = "STARTED"
    OK = "OK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"


class DomainStatus(StrEnum):
    OK = "OK"
    WARN = "WARN"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    NO_DATA = "NO_DATA"
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNKNOWN = "UNKNOWN"


class EvidenceStatus(StrEnum):
    VALID = "VALID"
    WEAK = "WEAK"
    INVALID = "INVALID"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class RecommendationStatus(StrEnum):
    ALLOW_RECOMMEND = "ALLOW_RECOMMEND"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DO_NOT_RECOMMEND = "DO_NOT_RECOMMEND"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


STRUCTURED_STATUS_KEYS = {
    "execution_status",
    "domain_status",
    "evidence_status",
    "recommendation_status",
}


def normalize_tool_result(
    tool_name: str,
    payload: dict[str, Any],
    *,
    execution_status: ExecutionStatus,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Return a payload with explicit status fields and raw result preserved."""

    raw_payload = dict(payload)
    explicit = STRUCTURED_STATUS_KEYS.issubset(payload.keys())
    raw_status = _status_text(payload.get("status"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata_status = _status_text(metadata.get("status")) if metadata else None
    diagnostic_status = _diagnostic_status(payload)

    domain_status = _enum_value(
        DomainStatus,
        payload.get("domain_status"),
        default=None,
    )
    evidence_status = _enum_value(
        EvidenceStatus,
        payload.get("evidence_status"),
        default=None,
    )
    recommendation_status = _enum_value(
        RecommendationStatus,
        payload.get("recommendation_status"),
        default=None,
    )

    warnings = _list_of_strings(payload.get("warnings"))
    blockers = _list_of_strings(payload.get("blockers"))

    if not explicit:
        inferred = _transparent_status_from_payload(
            raw_status=raw_status,
            metadata_status=metadata_status,
            diagnostic_status=diagnostic_status,
            payload=payload,
        )
        domain_status = domain_status or inferred["domain_status"]
        evidence_status = evidence_status or inferred["evidence_status"]
        recommendation_status = recommendation_status or inferred["recommendation_status"]
        if _is_legacy_unstructured(
            raw_status=raw_status,
            metadata_status=metadata_status,
            diagnostic_status=diagnostic_status,
        ):
            warnings.append("legacy_unstructured_tool_result")

    domain_status = domain_status or DomainStatus.UNKNOWN.value
    evidence_status = evidence_status or EvidenceStatus.UNKNOWN.value
    recommendation_status = recommendation_status or RecommendationStatus.UNKNOWN.value

    if domain_status in {DomainStatus.BLOCKED.value, DomainStatus.FAILED.value}:
        reason = payload.get("reason")
        if reason is None and metadata:
            reason = metadata.get("reason")
        if reason:
            blockers.append(str(reason))

    message = payload.get("message")
    if message is None and metadata:
        message = metadata.get("message")
    reason = payload.get("reason")
    if reason is None and metadata:
        reason = metadata.get("reason")
    next_repair_tool = payload.get("next_repair_tool")
    if next_repair_tool is None and metadata:
        next_repair_tool = metadata.get("next_repair_tool")

    normalized = dict(payload)
    normalized.update(
        {
            "tool_name": tool_name,
            "execution_status": execution_status.value,
            "domain_status": domain_status,
            "evidence_status": evidence_status,
            "recommendation_status": recommendation_status,
            "raw_status": raw_status,
            "diagnostic_status": diagnostic_status,
            "message": message,
            "reason": reason,
            "blockers": _dedupe(blockers),
            "warnings": _dedupe(warnings),
            "next_repair_tool": next_repair_tool,
            "suggested_repair": payload.get("suggested_repair"),
            "result": raw_payload,
        }
    )
    if duration_ms is not None:
        normalized["duration_ms"] = duration_ms
    return normalized


def audit_status_from_result(result: dict[str, Any]) -> str:
    execution = str(result.get("execution_status") or ExecutionStatus.OK.value).lower()
    domain = str(result.get("domain_status") or DomainStatus.UNKNOWN.value).lower()
    evidence = str(result.get("evidence_status") or EvidenceStatus.UNKNOWN.value).lower()
    if execution != ExecutionStatus.OK.value.lower():
        return f"execution_{execution}"
    if domain == DomainStatus.FAILED.value.lower():
        return "execution_ok_domain_failed"
    if domain == DomainStatus.BLOCKED.value.lower():
        return "execution_ok_domain_blocked"
    if domain == DomainStatus.PARTIAL.value.lower():
        return "execution_ok_domain_partial"
    if domain == DomainStatus.NO_DATA.value.lower():
        return "execution_ok_domain_no_data"
    if domain == DomainStatus.UNKNOWN.value.lower():
        return "execution_ok_domain_unknown"
    if evidence == EvidenceStatus.INVALID.value.lower():
        return "execution_ok_evidence_invalid"
    if evidence == EvidenceStatus.BLOCKED.value.lower():
        return "execution_ok_evidence_blocked"
    if evidence == EvidenceStatus.UNKNOWN.value.lower():
        return "execution_ok_evidence_unknown"
    return "execution_ok"


def status_icon(result: dict[str, Any]) -> str:
    execution = str(result.get("execution_status") or "")
    domain = str(result.get("domain_status") or "")
    evidence = str(result.get("evidence_status") or "")
    execution_failures = {
        ExecutionStatus.ERROR.value,
        ExecutionStatus.TIMEOUT.value,
        ExecutionStatus.PERMISSION_DENIED.value,
    }
    if execution in execution_failures:
        return "x"
    if domain == DomainStatus.BLOCKED.value or evidence == EvidenceStatus.BLOCKED.value:
        return "blocked"
    if domain == DomainStatus.FAILED.value or evidence == EvidenceStatus.INVALID.value:
        return "failed"
    if (
        domain in {DomainStatus.WARN.value, DomainStatus.PARTIAL.value}
        or evidence == EvidenceStatus.WEAK.value
    ):
        return "warning"
    if domain == DomainStatus.UNKNOWN.value or evidence == EvidenceStatus.UNKNOWN.value:
        return "unknown"
    return "ok"


def _transparent_status_from_payload(
    *,
    raw_status: str | None,
    metadata_status: str | None,
    diagnostic_status: str | None,
    payload: dict[str, Any],
) -> dict[str, str]:
    status = (raw_status or metadata_status or "").upper()
    if diagnostic_status == "FAIL":
        return {
            "domain_status": DomainStatus.FAILED.value,
            "evidence_status": EvidenceStatus.INVALID.value,
            "recommendation_status": RecommendationStatus.DO_NOT_RECOMMEND.value,
        }
    if status in {"BLOCKED", "MISSING_FACTOR_INPUTS"}:
        return {
            "domain_status": DomainStatus.BLOCKED.value,
            "evidence_status": EvidenceStatus.BLOCKED.value,
            "recommendation_status": RecommendationStatus.BLOCKED.value,
        }
    if status in {"NO_DATA", "NO_MATCHING_BARS"}:
        return {
            "domain_status": DomainStatus.NO_DATA.value,
            "evidence_status": EvidenceStatus.INCOMPLETE.value,
            "recommendation_status": RecommendationStatus.BLOCKED.value,
        }
    if status in {"PARTIAL_COVERAGE", "PARTIAL"}:
        return {
            "domain_status": DomainStatus.PARTIAL.value,
            "evidence_status": EvidenceStatus.INCOMPLETE.value,
            "recommendation_status": RecommendationStatus.UNKNOWN.value,
        }
    if status in {
        "INVALID_REQUEST",
        "STATIC_CHECK_FAILED",
        "SAMPLE_TEST_FAILED",
        "FACTOR_NOT_FOUND",
        "STRATEGY_NOT_FOUND",
        "BACKTEST_FAILED",
        "DATA_NOT_READY",
    }:
        return {
            "domain_status": DomainStatus.INVALID_REQUEST.value
            if status == "INVALID_REQUEST"
            else DomainStatus.FAILED.value,
            "evidence_status": EvidenceStatus.INVALID.value,
            "recommendation_status": RecommendationStatus.BLOCKED.value,
        }
    if status in {"NOT_AVAILABLE", "NOT_CONFIGURED"}:
        return {
            "domain_status": DomainStatus.NOT_CONFIGURED.value,
            "evidence_status": EvidenceStatus.BLOCKED.value,
            "recommendation_status": RecommendationStatus.BLOCKED.value,
        }
    if _has_missing_or_stale_coverage(payload):
        return {
            "domain_status": DomainStatus.PARTIAL.value,
            "evidence_status": EvidenceStatus.INCOMPLETE.value,
            "recommendation_status": RecommendationStatus.UNKNOWN.value,
        }
    if status == "OK":
        return {
            "domain_status": DomainStatus.OK.value,
            "evidence_status": EvidenceStatus.VALID.value,
            "recommendation_status": RecommendationStatus.RESEARCH_ONLY.value,
        }
    return {
        "domain_status": DomainStatus.UNKNOWN.value,
        "evidence_status": EvidenceStatus.UNKNOWN.value,
        "recommendation_status": RecommendationStatus.UNKNOWN.value,
    }


def _diagnostic_status(payload: dict[str, Any]) -> str | None:
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        status = diagnostics.get("status")
        return _status_text(status)
    return None


def _status_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _enum_value(enum_type: type[StrEnum], value: Any, *, default: str | None) -> str | None:
    if value is None:
        return default
    try:
        return enum_type(str(value)).value
    except ValueError:
        return default


def _list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _is_legacy_unstructured(
    *,
    raw_status: str | None,
    metadata_status: str | None,
    diagnostic_status: str | None,
) -> bool:
    status = (raw_status or metadata_status or "").upper()
    if diagnostic_status or status in {
        "OK",
        "BLOCKED",
        "NO_DATA",
        "NO_MATCHING_BARS",
        "PARTIAL_COVERAGE",
        "PARTIAL",
        "INVALID_REQUEST",
        "STATIC_CHECK_FAILED",
        "SAMPLE_TEST_FAILED",
        "NOT_AVAILABLE",
        "NOT_CONFIGURED",
    }:
        return False
    return True


def _has_missing_or_stale_coverage(payload: dict[str, Any]) -> bool:
    raw_metadata = payload.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    for source in (payload, metadata):
        if (
            source.get("missing_symbols")
            or source.get("stale_symbols")
            or source.get("missing_ranges")
        ):
            return True
        if source.get("data_update_needed") is True:
            return True
    return False
