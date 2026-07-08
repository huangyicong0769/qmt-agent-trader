"""Evidence ledger and post-hoc output conflict reporting.

The ledger records tool evidence and consistency conflicts. It does not modify
tool results or LLM final answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    RecommendationStatus,
)


@dataclass
class EvidenceLedger:
    run_id: str
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def record_tool_result(self, tool_name: str, result: Any) -> None:
        if not isinstance(result, dict):
            self.tool_results.append(
                {
                    "tool_name": tool_name,
                    "domain_status": DomainStatus.UNKNOWN.value,
                    "evidence_status": EvidenceStatus.UNKNOWN.value,
                    "recommendation_status": RecommendationStatus.UNKNOWN.value,
                    "raw_result": result,
                }
            )
            self.warnings.append(f"{tool_name}: non_dict_tool_result")
            return

        entry = {
            "tool_name": tool_name,
            "execution_status": result.get("execution_status"),
            "domain_status": result.get("domain_status"),
            "evidence_status": result.get("evidence_status"),
            "recommendation_status": result.get("recommendation_status"),
            "raw_status": result.get("raw_status"),
            "diagnostic_status": result.get("diagnostic_status"),
            "reason": result.get("reason"),
            "message": result.get("message"),
            "coverage_status": result.get("coverage_status"),
            "dataset_results": result.get("dataset_results", []),
            "blockers": result.get("blockers", []),
            "warnings": result.get("warnings", []),
            "next_repair_tool": result.get("next_repair_tool"),
            "result_id_fields": _result_ref_fields(result),
        }
        self.tool_results.append(entry)
        self.blockers.extend(str(item) for item in result.get("blockers", []) if str(item))
        self.warnings.extend(str(item) for item in result.get("warnings", []) if str(item))
        self._record_status_conflicts(tool_name, result)

    def report(self) -> dict[str, Any]:
        summary = {
            "valid_count": 0,
            "weak_count": 0,
            "invalid_count": 0,
            "blocked_count": 0,
            "incomplete_count": 0,
            "unknown_count": 0,
        }
        for item in self.tool_results:
            evidence = str(item.get("evidence_status") or EvidenceStatus.UNKNOWN.value)
            if evidence == EvidenceStatus.VALID.value:
                summary["valid_count"] += 1
            elif evidence == EvidenceStatus.WEAK.value:
                summary["weak_count"] += 1
            elif evidence == EvidenceStatus.INVALID.value:
                summary["invalid_count"] += 1
            elif evidence == EvidenceStatus.BLOCKED.value:
                summary["blocked_count"] += 1
            elif evidence == EvidenceStatus.INCOMPLETE.value:
                summary["incomplete_count"] += 1
            else:
                summary["unknown_count"] += 1
        return {
            "run_id": self.run_id,
            "summary": summary,
            "conflicts": self.conflicts,
            "blockers": _dedupe(self.blockers),
            "warnings": _dedupe(self.warnings),
            "tool_results": self.tool_results,
        }

    def final_answer_conflict_report(self, final_answer_raw: str) -> dict[str, Any]:
        conflicts = list(self.conflicts)
        overclaim = _recommendation_excerpt(final_answer_raw)
        if overclaim:
            for item in self.tool_results:
                recommendation = str(item.get("recommendation_status") or "")
                evidence = str(item.get("evidence_status") or "")
                diagnostic = str(item.get("diagnostic_status") or "")
                do_not_recommend = {
                    RecommendationStatus.DO_NOT_RECOMMEND.value,
                    RecommendationStatus.BLOCKED.value,
                }
                if (
                    recommendation in do_not_recommend
                    or evidence in {EvidenceStatus.INVALID.value, EvidenceStatus.BLOCKED.value}
                    or diagnostic == "FAIL"
                ):
                    conflicts.append(
                        {
                            "type": "UNSUPPORTED_RECOMMENDATION",
                            "severity": "HIGH",
                            "answer_excerpt": overclaim,
                            "evidence_ref": item.get("tool_name"),
                            "message": (
                                "Final answer contains positive/recommendation language "
                                "while tool evidence is failed, blocked, or invalid."
                            ),
                        }
                    )
                    break
        coverage_overclaim = _coverage_overclaim_excerpt(final_answer_raw)
        if coverage_overclaim:
            for item in self.tool_results:
                incomplete = _incomplete_data_evidence(item)
                if incomplete:
                    conflicts.append(
                        {
                            "type": "DATA_COVERAGE_OVERCLAIM",
                            "severity": "HIGH",
                            "answer_excerpt": coverage_overclaim,
                            "evidence_ref": item.get("tool_name"),
                            "coverage_status": item.get("coverage_status"),
                            "dataset_results": incomplete,
                            "message": (
                                "Final answer claims data completion or successful "
                                "coverage while structured tool evidence is incomplete."
                            ),
                        }
                    )
                    break
        has_conflict = bool(conflicts)
        return {
            "has_conflict": has_conflict,
            "severity": _max_severity(conflicts),
            "final_answer_raw": final_answer_raw,
            "conflicts": conflicts,
        }

    def _record_status_conflicts(self, tool_name: str, result: dict[str, Any]) -> None:
        raw_status = str(result.get("raw_status") or "")
        diagnostic_status = str(result.get("diagnostic_status") or "")
        execution_status = str(result.get("execution_status") or "")
        domain_status = str(result.get("domain_status") or "")
        evidence_status = str(result.get("evidence_status") or "")
        if raw_status == "completed" and diagnostic_status == "FAIL":
            self.conflicts.append(
                {
                    "type": "COMPLETED_WITH_FAILED_DIAGNOSTICS",
                    "severity": "HIGH",
                    "tool_name": tool_name,
                    "message": "Tool completed computation but diagnostics.status=FAIL.",
                }
            )
        if execution_status == "OK" and domain_status in {
            DomainStatus.BLOCKED.value,
            DomainStatus.FAILED.value,
            DomainStatus.NO_DATA.value,
        }:
            self.conflicts.append(
                {
                    "type": "EXECUTION_OK_DOMAIN_NOT_OK",
                    "severity": "MEDIUM",
                    "tool_name": tool_name,
                    "domain_status": domain_status,
                    "message": "Python execution returned, but domain status is not OK.",
                }
            )
        if evidence_status in {EvidenceStatus.INVALID.value, EvidenceStatus.BLOCKED.value}:
            self.conflicts.append(
                {
                    "type": "EVIDENCE_NOT_USABLE",
                    "severity": "HIGH",
                    "tool_name": tool_name,
                    "evidence_status": evidence_status,
                    "message": "Tool evidence is invalid or blocked.",
                }
            )
        coverage_status = str(result.get("coverage_status") or "")
        if coverage_status in {"PARTIAL_COVERAGE", "NO_DATA", "INVALID_REQUEST", "BLOCKED"}:
            self.conflicts.append(
                {
                    "type": "DATA_COVERAGE_NOT_COMPLETE",
                    "severity": "MEDIUM",
                    "tool_name": tool_name,
                    "coverage_status": coverage_status,
                    "message": "Structured data coverage is not complete.",
                }
            )


def _recommendation_excerpt(text: str) -> str | None:
    patterns = [
        r".{0,20}(推荐|最有希望|最优|最佳|最好|有效|显著|稳健|核心发现).{0,40}",
        r".{0,20}(recommend|best|optimal|effective|robust).{0,40}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _coverage_overclaim_excerpt(text: str) -> str | None:
    patterns = [
        r".{0,20}(已补齐|补齐完成|数据完整|覆盖完整|完整覆盖|全部成功|成功更新|更新成功|3/3.*成功).{0,40}",
        r".{0,20}(complete|fully covered|all successful|successfully updated).{0,40}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _incomplete_data_evidence(item: dict[str, Any]) -> list[dict[str, Any]]:
    coverage_status = str(item.get("coverage_status") or "")
    incomplete_statuses = {
        "PARTIAL_COVERAGE",
        "NO_DATA",
        "NOT_VERIFIED",
        "INVALID_REQUEST",
        "BLOCKED",
    }
    dataset_results = item.get("dataset_results")
    incomplete_results: list[dict[str, Any]] = []
    if isinstance(dataset_results, list):
        for result in dataset_results:
            if not isinstance(result, dict):
                continue
            status = str(result.get("status") or "")
            rows = int(result.get("rows", 0) or 0)
            result_coverage = str(result.get("coverage_status") or "")
            if status != "updated" or rows <= 0 or result_coverage in incomplete_statuses:
                incomplete_results.append(
                    {
                        "dataset_id": result.get("dataset_id"),
                        "api_name": result.get("api_name"),
                        "status": status,
                        "rows": rows,
                        "coverage_status": result_coverage,
                        "reason": result.get("reason"),
                    }
                )
    if incomplete_results:
        return incomplete_results
    if coverage_status in incomplete_statuses:
        return [{"coverage_status": coverage_status}]
    return []


def _result_ref_fields(result: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("run_id", "strategy_id", "factor_id", "report_path", "code_path"):
        if result.get(key):
            fields[key] = result[key]
    return fields


def _max_severity(conflicts: list[dict[str, Any]]) -> str:
    if not conflicts:
        return "NONE"
    if any(item.get("severity") == "HIGH" for item in conflicts):
        return "HIGH"
    if any(item.get("severity") == "MEDIUM" for item in conflicts):
        return "MEDIUM"
    return "LOW"


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
