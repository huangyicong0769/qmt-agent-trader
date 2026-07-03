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
            return FetchResult(
                status=plan.status,
                source="tushare",
                errors=plan.errors,
                metadata=plan.as_dict(),
            )
        if dry_run or not execute_plan:
            return FetchResult(
                status="planned",
                source="tushare",
                metadata={**plan.as_dict(), "dry_run": dry_run, "execute_plan": execute_plan},
            )

        writes: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for item in plan.items:
            spec = self.registry.require(str(item["api_name"]))
            frames: list[pd.DataFrame] = []
            for batch in item.get("batches", []):
                frame, page_errors = self._execute_batch(batch)
                errors.extend(page_errors)
                if not frame.empty:
                    frames.append(frame)
            result_frame = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=item["fields"])
            )
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
                self._record_metadata(item, status="SCHEMA_MISMATCH", row_count=0, error=error)
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
            writes.append(
                {
                    "dataset_id": spec.dataset_id,
                    "layer": "raw",
                    "path": str(path),
                    "view": spec.raw_view_name,
                    "rows": len(result_frame),
                }
            )
        status = "updated" if writes and not errors else "PARTIAL_UPDATE" if writes else "error"
        return FetchResult(
            status=status,
            source="tushare",
            writes=writes,
            errors=errors,
            metadata={"coverage_status": "OK" if not errors else "PARTIAL_COVERAGE"},
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
