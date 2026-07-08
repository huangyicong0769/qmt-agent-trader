"""Execute validated Tushare fetch plans."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from qmt_agent_trader.data.providers.base import FetchPlan, FetchResult
from qmt_agent_trader.data.providers.tushare.client import TushareClient
from qmt_agent_trader.data.providers.tushare.registry import (
    TushareEndpointRegistry,
    default_tushare_registry,
)
from qmt_agent_trader.data.storage import DataLake


class TushareFetcher:
    def __init__(
        self,
        client: TushareClient,
        lake: DataLake,
        *,
        registry: TushareEndpointRegistry | None = None,
        min_interval_seconds: float = 0.3,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.lake = lake
        self.registry = registry or default_tushare_registry()
        self.min_interval_seconds = min_interval_seconds
        self.retry_attempts = max(retry_attempts, 1)
        self.retry_backoff_seconds = retry_backoff_seconds
        self.sleep = sleep
        self._last_request_at: float | None = None

    def run(self, plan: FetchPlan, *, execute_plan: bool, dry_run: bool = False) -> FetchResult:
        if plan.status != "planned":
            plan_payload = plan.as_dict()
            return FetchResult(
                status=plan.status,
                source="tushare",
                errors=plan.errors,
                metadata=plan_payload,
                execution_status=plan_payload["execution_status"],
                domain_status=plan_payload["domain_status"],
                evidence_status=plan_payload["evidence_status"],
                recommendation_status=plan_payload["recommendation_status"],
                coverage_status=plan_payload["coverage_status"],
                warnings=plan.warnings,
                blockers=plan.blockers,
                next_repair_tool=plan.next_repair_tool,
                suggested_repair=plan.suggested_repair,
                repair_action=plan.repair_action,
                verification_action=plan.verification_action,
            )
        if dry_run or not execute_plan:
            return FetchResult(
                status="planned",
                source="tushare",
                metadata={**plan.as_dict(), "dry_run": dry_run, "execute_plan": execute_plan},
                execution_status="OK",
                domain_status="OK",
                evidence_status="WEAK",
                recommendation_status="RESEARCH_ONLY",
                coverage_status="NOT_VERIFIED",
                verification_action=_verification_action_for_items(plan.items),
            )

        writes: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        dataset_results: list[dict[str, Any]] = []
        warnings: list[str] = []
        for item in plan.items:
            spec = self.registry.require(str(item["api_name"]))
            frames: list[pd.DataFrame] = []
            item_errors: list[dict[str, Any]] = []
            for batch in item.get("batches", []):
                try:
                    frame, page_errors = self._execute_batch(batch)
                except Exception as exc:
                    error = {
                        "status": "FAILED",
                        "api_name": spec.api_name,
                        "dataset_id": spec.dataset_id,
                        "reason": "remote_query_failed",
                        "message": str(exc),
                    }
                    item_errors.append(error)
                    errors.append(error)
                    continue
                item_errors.extend(page_errors)
                errors.extend(page_errors)
                if not frame.empty:
                    frames.append(frame)
            result_frame = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=item["fields"])
            )
            if result_frame.empty and not item_errors:
                dataset_result = _dataset_result(
                    item,
                    api_name=spec.api_name,
                    status="NO_DATA",
                    rows=0,
                    coverage_status="NO_DATA",
                    reason="zero_rows_returned",
                    write_skipped=True,
                )
                dataset_results.append(dataset_result)
                warnings.append(f"zero_rows_for_dataset:{spec.dataset_id}")
                self._record_metadata(
                    item,
                    status="NO_DATA",
                    row_count=0,
                    checksum=None,
                    error={"reason": "zero_rows_returned"},
                )
                continue
            if result_frame.empty and item_errors:
                dataset_results.append(
                    _dataset_result(
                        item,
                        api_name=spec.api_name,
                        status="FAILED",
                        rows=0,
                        coverage_status="INVALID_REQUEST",
                        reason="remote_query_failed",
                        write_skipped=True,
                        errors=item_errors,
                    )
                )
                self._record_metadata(
                    item,
                    status="FAILED",
                    row_count=0,
                    checksum=None,
                    error={"errors": item_errors},
                )
                continue
            missing_columns = [
                column for column in spec.key_columns if column not in result_frame.columns
            ]
            requested_missing = [
                column for column in item["fields"] if column not in result_frame.columns
            ]
            if missing_columns or requested_missing:
                error = {
                    "status": "SCHEMA_MISMATCH",
                    "api_name": spec.api_name,
                    "missing_columns": sorted(set(missing_columns + requested_missing)),
                    "message": "Tushare response missing required columns; write skipped.",
                }
                errors.append(error)
                dataset_results.append(
                    _dataset_result(
                        item,
                        api_name=spec.api_name,
                        status="SCHEMA_MISMATCH",
                        rows=0,
                        coverage_status="INVALID_REQUEST",
                        reason="schema_mismatch",
                        write_skipped=True,
                        errors=[error],
                    )
                )
                self._record_metadata(
                    item,
                    status="SCHEMA_MISMATCH",
                    row_count=0,
                    checksum=None,
                    error=error,
                )
                continue
            path = self.lake.write_incremental_dataset(
                result_frame,
                layer="raw",
                dataset_id=spec.dataset_id,
                name=spec.raw_dataset_name,
                key_columns=list(spec.key_columns),
            )
            checksum = _checksum_frame(result_frame)
            self._record_metadata(
                item,
                status="success",
                row_count=len(result_frame),
                checksum=checksum,
                error=None,
            )
            write = {
                "dataset_id": spec.dataset_id,
                "layer": "raw",
                "path": str(path),
                "view": spec.raw_view_name,
                "rows": len(result_frame),
            }
            writes.append(write)
            dataset_results.append(
                {
                    **_dataset_result(
                        item,
                        api_name=spec.api_name,
                        status="updated",
                        rows=len(result_frame),
                        coverage_status="OK",
                        reason=None,
                        write_skipped=False,
                    ),
                    "path": str(path),
                    "view": spec.raw_view_name,
                }
            )
        outcome = _aggregate_fetch_outcome(dataset_results, errors)
        return FetchResult(
            status=outcome["status"],
            source="tushare",
            writes=writes,
            dataset_results=dataset_results,
            errors=errors,
            metadata={
                "coverage_status": outcome["coverage_status"],
                "total_rows_written": sum(int(write.get("rows", 0)) for write in writes),
                "dataset_count": len(dataset_results),
                "updated_dataset_count": sum(
                    1
                    for item in dataset_results
                    if item.get("status") == "updated" and int(item.get("rows", 0)) > 0
                ),
            },
            execution_status="OK",
            domain_status=outcome["domain_status"],
            evidence_status=outcome["evidence_status"],
            recommendation_status=outcome["recommendation_status"],
            coverage_status=outcome["coverage_status"],
            warnings=warnings,
            blockers=outcome["blockers"],
            next_repair_tool=outcome["next_repair_tool"],
            verification_action=_verification_action_for_items(plan.items),
        )

    def _execute_batch(self, batch: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        pagination = batch.get("pagination") or {}
        if pagination.get("type") != "limit_offset":
            return self._query(batch), []

        page_size = int(pagination.get("page_size", 5000))
        max_pages = int(pagination.get("max_pages", 20))
        limit_param = str(pagination.get("limit_param", "limit"))
        offset_param = str(pagination.get("offset_param", "offset"))
        frames: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        previous: pd.DataFrame | None = None
        for page_index in range(max_pages):
            page_batch = dict(batch)
            params = dict(batch["params"])
            params[limit_param] = page_size
            params[offset_param] = page_index * page_size
            page_batch["params"] = params
            frame = self._query(page_batch)
            if frame.empty:
                break
            if previous is not None and frame.reset_index(drop=True).equals(
                previous.reset_index(drop=True)
            ):
                errors.append(
                    {
                        "status": "PARTIAL_UPDATE",
                        "reason": "duplicate_pagination_page",
                        "api_name": batch["api_name"],
                    }
                )
                break
            frames.append(frame)
            if len(frame) < page_size:
                break
            previous = frame
        else:
            errors.append(
                {
                    "status": "PARTIAL_UPDATE",
                    "reason": "pagination_page_limit_exceeded",
                    "api_name": batch["api_name"],
                    "max_pages": max_pages,
                }
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), errors

    def _query(self, batch: dict[str, Any]) -> pd.DataFrame:
        for attempt in range(1, self.retry_attempts + 1):
            self._wait()
            try:
                return self.client.query(
                    str(batch["api_name"]),
                    dict(batch["params"]),
                    list(batch["fields"]),
                )
            except Exception:
                if attempt >= self.retry_attempts:
                    raise
                if self.retry_backoff_seconds > 0:
                    self.sleep(self.retry_backoff_seconds * attempt)
        return pd.DataFrame()

    def _wait(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = self.min_interval_seconds - (now - self._last_request_at)
            if remaining > 0:
                self.sleep(remaining)
                now = time.monotonic()
        self._last_request_at = now

    def _record_metadata(
        self,
        item: dict[str, Any],
        *,
        status: str,
        row_count: int,
        error: dict[str, Any] | None,
        checksum: str | None = None,
    ) -> None:
        params = dict(item.get("params", {}))
        coverage_start, coverage_end = _coverage_bounds_from_params(params)
        self.lake.record_fetch_metadata(
            source="tushare",
            dataset_id=str(item["dataset_id"]),
            api_name=str(item["api_name"]),
            endpoint_id=str(item["api_name"]),
            params=params,
            fields=list(item.get("fields", [])),
            symbols=list(item.get("symbols", [])),
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            row_count=row_count,
            checksum=checksum,
            status=status,
            error=json.dumps(error, ensure_ascii=True, sort_keys=True) if error else None,
        )


def _checksum_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    normalized = frame.sort_index(axis=1).astype(str)
    payload = normalized.to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _dataset_result(
    item: dict[str, Any],
    *,
    api_name: str,
    status: str,
    rows: int,
    coverage_status: str,
    reason: str | None,
    write_skipped: bool,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    params = dict(item.get("params", {}))
    requested_start, requested_end = _coverage_bounds_from_params(params)
    return {
        "dataset_id": str(item["dataset_id"]),
        "api_name": api_name,
        "status": status,
        "rows": rows,
        "requested_symbols": list(item.get("symbols", [])),
        "requested_start_date": requested_start,
        "requested_end_date": requested_end,
        "coverage_status": coverage_status,
        "reason": reason,
        "write_skipped": write_skipped,
        "errors": errors or [],
    }


def _aggregate_fetch_outcome(
    dataset_results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = [
        item
        for item in dataset_results
        if item.get("status") == "updated" and int(item.get("rows", 0)) > 0
    ]
    statuses = {str(item.get("status")) for item in dataset_results}
    blockers: list[str] = []
    if any(status == "SCHEMA_MISMATCH" for status in statuses):
        blockers.append("schema_mismatch")
        if not updated:
            return {
                "status": "SCHEMA_MISMATCH",
                "domain_status": "FAILED",
                "evidence_status": "INVALID",
                "recommendation_status": "BLOCKED",
                "coverage_status": "INVALID_REQUEST",
                "blockers": blockers,
                "next_repair_tool": "list_tushare_capabilities",
            }
    if not updated and statuses == {"NO_DATA"}:
        return {
            "status": "NO_DATA",
            "domain_status": "NO_DATA",
            "evidence_status": "INCOMPLETE",
            "recommendation_status": "BLOCKED",
            "coverage_status": "NO_DATA",
            "blockers": ["zero_rows_returned"],
            "next_repair_tool": None,
        }
    if updated and (
        errors
        or any(status != "updated" for status in statuses)
        or len(updated) != len(dataset_results)
    ):
        return {
            "status": "PARTIAL_UPDATE",
            "domain_status": "PARTIAL",
            "evidence_status": "INCOMPLETE",
            "recommendation_status": "UNKNOWN",
            "coverage_status": "PARTIAL_COVERAGE",
            "blockers": blockers,
            "next_repair_tool": None,
        }
    if updated and not errors and len(updated) == len(dataset_results):
        return {
            "status": "updated",
            "domain_status": "OK",
            "evidence_status": "VALID",
            "recommendation_status": "RESEARCH_ONLY",
            "coverage_status": "OK",
            "blockers": [],
            "next_repair_tool": None,
        }
    return {
        "status": "FAILED",
        "domain_status": "FAILED",
        "evidence_status": "INVALID",
        "recommendation_status": "BLOCKED",
        "coverage_status": "INVALID_REQUEST",
        "blockers": blockers or ["remote_fetch_failed"],
        "next_repair_tool": "list_tushare_capabilities",
    }


def _verification_action_for_items(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    api_names = {str(item.get("api_name")) for item in items}
    symbols = _dedupe_symbols(
        symbol for item in items for symbol in list(item.get("symbols", []))
    )
    start, end = _combined_bounds(items)
    if api_names.issubset({"daily", "fund_daily", "index_daily"}):
        return {
            "tool": "query_bars",
            "input": {
                "symbols": symbols,
                "start_date": start,
                "end_date": end,
            },
        }
    if api_names.intersection(
        {"daily_basic", "income", "balancesheet", "cashflow", "fina_indicator", "dividend"}
    ):
        return {
            "tool": "query_fundamentals_pit",
            "input": {
                "symbols": symbols,
                "as_of_date": end or start,
                "include_daily_basic": "daily_basic" in api_names,
                "include_financials": bool(
                    api_names.intersection(
                        {"income", "balancesheet", "cashflow", "fina_indicator", "dividend"}
                    )
                ),
            },
        }
    if len(api_names) == 1:
        api_name = next(iter(api_names))
        if api_name.startswith("cn_") or api_name == "shibor":
            return {
                "tool": "query_macro_series_pit",
                "input": {
                    "dataset": api_name,
                    "start_date": start,
                    "end_date": end,
                    "as_of_date": end or start,
                },
            }
    return None


def _combined_bounds(items: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    starts: list[str] = []
    ends: list[str] = []
    for item in items:
        start, end = _coverage_bounds_from_params(dict(item.get("params", {})))
        if start:
            starts.append(start)
        if end:
            ends.append(end)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _dedupe_symbols(symbols: Any) -> list[str]:
    result: list[str] = []
    for symbol in symbols:
        text = str(symbol)
        if text and text not in result:
            result.append(text)
    return result


def _coverage_bounds_from_params(params: dict[str, Any]) -> tuple[str | None, str | None]:
    for start_key, end_key in (
        ("start_date", "end_date"),
        ("start_m", "end_m"),
        ("start_q", "end_q"),
    ):
        start = _optional_text(params.get(start_key))
        end = _optional_text(params.get(end_key))
        if start or end:
            return start, end
    for point_key in ("trade_date", "date", "cal_date", "m", "q", "period", "ann_date"):
        value = _optional_text(params.get(point_key))
        if value:
            return value, value
    return None, None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
