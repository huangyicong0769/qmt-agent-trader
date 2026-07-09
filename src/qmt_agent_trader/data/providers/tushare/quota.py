"""Quota and usage accounting for Tushare data fetches."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

QuotaSource = Literal[
    "official_table",
    "manual_config",
    "api",
    "observed",
    "unknown",
    "fallback_static_policy",
]
QuotaScope = Literal["account_global", "per_api", "unknown"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
ExecutionMode = Literal["autonomous", "approved", "manual", "dry_run"]
UsageStatus = Literal[
    "PLANNED",
    "SUCCESS",
    "NO_DATA",
    "FAILED",
    "RATE_LIMITED",
    "QUOTA_EXCEEDED",
    "SKIPPED_LOCAL_CACHE",
]
DecisionStatus = Literal[
    "APPROVED_BY_ACCOUNT_QUOTA",
    "APPROVED_WITH_RATE_PACING",
    "NEEDS_USER_APPROVAL",
    "WAIT_FOR_RATE_LIMIT_RESET",
    "BLOCKED_BY_DAILY_QUOTA",
    "BLOCKED_BY_ENDPOINT_PERMISSION",
    "UNKNOWN_QUOTA_REQUIRES_APPROVAL",
]

EXECUTED_USAGE_STATUSES: set[str] = {
    "SUCCESS",
    "NO_DATA",
    "FAILED",
    "RATE_LIMITED",
    "QUOTA_EXCEEDED",
}
COMPLETED_CACHE_STATUSES: set[str] = {"SUCCESS", "NO_DATA"}


class TushareAccountQuotaProfile(BaseModel):
    source: QuotaSource
    points: int | None = None
    tier: str | None = None
    max_requests_per_minute: int | None = None
    max_requests_per_day_per_api: int | None = None
    minute_quota_scope: QuotaScope = "unknown"
    daily_quota_scope: QuotaScope = "unknown"
    confidence: Confidence
    updated_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)


DEFAULT_TUSHARE_2000_POINT_PROFILE = TushareAccountQuotaProfile(
    source="official_table",
    points=2000,
    tier="2000+",
    max_requests_per_minute=200,
    max_requests_per_day_per_api=100000,
    minute_quota_scope="account_global",
    daily_quota_scope="per_api",
    confidence="MEDIUM",
    notes=[
        "Default project profile configured from Tushare official point-frequency table."
    ],
)


class TushareUsageRecord(BaseModel):
    request_id: str
    run_id: str | None = None
    api_name: str
    params_hash: str
    params_redacted: dict[str, Any]
    fields: list[str]
    planned_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: UsageStatus
    row_count: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    token_hash: str | None = None
    execution_mode: ExecutionMode


class TushareQuotaState(BaseModel):
    profile: TushareAccountQuotaProfile
    used_requests_last_minute: int
    remaining_requests_this_minute: int | None
    used_requests_today_by_api: dict[str, int]
    remaining_requests_today_by_api: dict[str, int | None]
    recent_rate_limit_errors: list[str]
    confidence: Confidence


class EndpointCostBreakdown(BaseModel):
    api_name: str
    planned_batches: int
    local_cache_hits: int
    net_new_batches: int
    estimated_request_count: int
    estimated_cost_units: int
    missing_time_points: list[str] = Field(default_factory=list)
    missing_symbols: list[str] = Field(default_factory=list)


class TushareFetchCostEstimate(BaseModel):
    estimated_request_count: int
    net_new_request_count: int
    endpoint_breakdown: dict[str, EndpointCostBreakdown]
    local_cache_hits: int
    remote_fetch_required: int
    estimated_duration_seconds: float | None = None
    assumptions: list[str] = Field(default_factory=list)


class TushareBudgetDecision(BaseModel):
    status: DecisionStatus
    reason: str
    estimated_request_count: int
    net_new_request_count: int
    quota_profile: TushareAccountQuotaProfile
    quota_state: TushareQuotaState
    safe_to_execute_now: bool
    recommended_batch_size: int | None = None
    recommended_batches: list[dict[str, Any]] = Field(default_factory=list)
    user_message: str


class TushareUsageLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_lake_root(cls, lake_root: Path) -> TushareUsageLedger:
        return cls(lake_root / "metadata" / "tushare_usage_ledger.parquet")

    def append(self, record: TushareUsageRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([_record_to_row(record)])
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_parquet(self.path, index=False)

    def usage_last_minute(self, *, now: datetime | None = None) -> int:
        cutoff = _as_naive_utc(now) - timedelta(minutes=1)
        frame = self._read()
        if frame.empty:
            return 0
        finished_at = _datetime_series(frame["finished_at"])
        return int(
            frame[
                frame["status"].isin(EXECUTED_USAGE_STATUSES)
                & (finished_at >= cutoff)
            ].shape[0]
        )

    def usage_today_by_api(self, *, now: datetime | None = None) -> dict[str, int]:
        today = _as_naive_utc(now).date()
        frame = self._read()
        if frame.empty:
            return {}
        finished = _datetime_series(frame["finished_at"])
        scoped = frame[
            frame["status"].isin(EXECUTED_USAGE_STATUSES)
            & (finished.dt.date == today)
        ]
        if scoped.empty:
            return {}
        counts = scoped.groupby("api_name").size().to_dict()
        return {str(api_name): int(count) for api_name, count in counts.items()}

    def usage_today(self, api_name: str, *, now: datetime | None = None) -> int:
        return self.usage_today_by_api(now=now).get(api_name, 0)

    def recent_rate_limit_errors(
        self,
        *,
        minutes: int = 10,
    ) -> list[TushareUsageRecord]:
        cutoff = _as_naive_utc(None) - timedelta(minutes=minutes)
        frame = self._read()
        if frame.empty:
            return []
        scoped = frame[
            (frame["status"] == "RATE_LIMITED")
            & (_datetime_series(frame["finished_at"]) >= cutoff)
        ]
        return [_row_to_record(row) for row in scoped.to_dict(orient="records")]

    def request_seen(self, api_name: str, params_hash: str) -> bool:
        frame = self._read()
        if frame.empty:
            return False
        scoped = frame[
            (frame["api_name"] == api_name)
            & (frame["params_hash"] == params_hash)
            & frame["status"].isin(COMPLETED_CACHE_STATUSES)
        ]
        return not scoped.empty

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            return _empty_ledger_frame()
        return pd.read_parquet(self.path)


class TushareQuotaManager:
    def __init__(
        self,
        *,
        profile: TushareAccountQuotaProfile,
        ledger: TushareUsageLedger | None = None,
    ) -> None:
        self.profile = profile
        self.ledger = ledger

    def current_state(self) -> TushareQuotaState:
        used_last_minute = (
            self.ledger.usage_last_minute() if self.ledger is not None else 0
        )
        used_today = self.ledger.usage_today_by_api() if self.ledger is not None else {}
        remaining_minute = (
            max(self.profile.max_requests_per_minute - used_last_minute, 0)
            if self.profile.max_requests_per_minute is not None
            else None
        )
        remaining_by_api: dict[str, int | None] = {}
        if self.profile.max_requests_per_day_per_api is not None:
            for api_name, used in used_today.items():
                remaining_by_api[api_name] = max(
                    self.profile.max_requests_per_day_per_api - used,
                    0,
                )
        recent_errors = (
            [
                f"{record.api_name}:{record.error_type or record.status}"
                for record in self.ledger.recent_rate_limit_errors()
            ]
            if self.ledger is not None
            else []
        )
        return TushareQuotaState(
            profile=self.profile,
            used_requests_last_minute=used_last_minute,
            remaining_requests_this_minute=remaining_minute,
            used_requests_today_by_api=used_today,
            remaining_requests_today_by_api=remaining_by_api,
            recent_rate_limit_errors=recent_errors,
            confidence=self.profile.confidence,
        )

    def estimate_cost(
        self,
        planned_items: list[dict[str, Any]],
        *,
        local_cache_state: dict[str, Any] | None = None,
    ) -> TushareFetchCostEstimate:
        del local_cache_state
        breakdown: dict[str, EndpointCostBreakdown] = {}
        estimated = 0
        local_hits = 0
        net_new = 0
        for item in planned_items:
            api_name = str(item["api_name"])
            batches = cast(list[dict[str, Any]], item.get("batches", []))
            planned_batches = len(batches)
            endpoint_hits = 0
            endpoint_net_new = 0
            for batch in batches:
                params_hash = normalized_request_hash(
                    api_name=api_name,
                    params=cast(dict[str, Any], batch.get("params", {})),
                    fields=[str(field) for field in batch.get("fields", [])],
                )
                if self.ledger is not None and self.ledger.request_seen(api_name, params_hash):
                    endpoint_hits += 1
                else:
                    endpoint_net_new += 1
            estimated += planned_batches
            local_hits += endpoint_hits
            net_new += endpoint_net_new
            existing = breakdown.get(api_name)
            if existing is None:
                breakdown[api_name] = EndpointCostBreakdown(
                    api_name=api_name,
                    planned_batches=planned_batches,
                    local_cache_hits=endpoint_hits,
                    net_new_batches=endpoint_net_new,
                    estimated_request_count=planned_batches,
                    estimated_cost_units=endpoint_net_new,
                )
            else:
                breakdown[api_name] = existing.model_copy(
                    update={
                        "planned_batches": existing.planned_batches + planned_batches,
                        "local_cache_hits": existing.local_cache_hits + endpoint_hits,
                        "net_new_batches": existing.net_new_batches + endpoint_net_new,
                        "estimated_request_count": (
                            existing.estimated_request_count + planned_batches
                        ),
                        "estimated_cost_units": existing.estimated_cost_units
                        + endpoint_net_new,
                    }
                )
        return TushareFetchCostEstimate(
            estimated_request_count=estimated,
            net_new_request_count=net_new,
            endpoint_breakdown=breakdown,
            local_cache_hits=local_hits,
            remote_fetch_required=net_new,
            assumptions=[
                "Net-new request count excludes successful equivalent requests "
                "already seen in the usage ledger."
            ],
        )

    def evaluate(
        self,
        cost: TushareFetchCostEstimate,
        state: TushareQuotaState,
        *,
        execution_mode: ExecutionMode,
    ) -> TushareBudgetDecision:
        if execution_mode == "dry_run":
            return _decision(
                "APPROVED_BY_ACCOUNT_QUOTA",
                "Dry run does not consume Tushare quota.",
                cost,
                state,
                safe_to_execute_now=True,
            )
        if (
            state.profile.max_requests_per_minute is None
            or state.profile.max_requests_per_day_per_api is None
            or state.profile.source == "unknown"
        ):
            return _decision(
                "UNKNOWN_QUOTA_REQUIRES_APPROVAL",
                "Tushare quota profile is unknown; user approval is required "
                "before live execution.",
                cost,
                state,
                safe_to_execute_now=False,
            )
        daily_blockers: list[str] = []
        for api_name, endpoint_cost in cost.endpoint_breakdown.items():
            used_today = state.used_requests_today_by_api.get(api_name, 0)
            remaining = state.profile.max_requests_per_day_per_api - used_today
            if endpoint_cost.net_new_batches > remaining:
                daily_blockers.append(
                    f"{api_name} requires {endpoint_cost.net_new_batches} net-new requests, "
                    f"but only {remaining} daily requests remain."
                )
        if daily_blockers:
            return _decision(
                "BLOCKED_BY_DAILY_QUOTA",
                " ".join(daily_blockers),
                cost,
                state,
                safe_to_execute_now=False,
            )
        remaining_minute = state.remaining_requests_this_minute
        if remaining_minute is not None and cost.net_new_request_count > remaining_minute:
            return _decision(
                "APPROVED_WITH_RATE_PACING",
                (
                    f"{cost.net_new_request_count} net-new requests fit the daily quota but exceed "
                    f"the current minute remainder of {remaining_minute}."
                ),
                cost,
                state,
                safe_to_execute_now=True,
                recommended_batch_size=max(remaining_minute, 1),
                recommended_batches=_recommended_batches(
                    cost.net_new_request_count,
                    max(remaining_minute, 1),
                ),
            )
        return _decision(
            "APPROVED_BY_ACCOUNT_QUOTA",
            (
                f"{cost.net_new_request_count} net-new requests are within the "
                f"{state.profile.tier or 'configured'} Tushare account quota."
            ),
            cost,
            state,
            safe_to_execute_now=True,
        )


def profile_from_settings(
    *,
    source: QuotaSource,
    points: int | None,
    max_requests_per_minute: int | None,
    max_requests_per_day_per_api: int | None,
) -> TushareAccountQuotaProfile:
    if (
        source == DEFAULT_TUSHARE_2000_POINT_PROFILE.source
        and points == DEFAULT_TUSHARE_2000_POINT_PROFILE.points
        and max_requests_per_minute
        == DEFAULT_TUSHARE_2000_POINT_PROFILE.max_requests_per_minute
        and max_requests_per_day_per_api
        == DEFAULT_TUSHARE_2000_POINT_PROFILE.max_requests_per_day_per_api
    ):
        return DEFAULT_TUSHARE_2000_POINT_PROFILE.model_copy(deep=True)
    tier = f"{points}+" if points is not None else None
    return TushareAccountQuotaProfile(
        source=source,
        points=points,
        tier=tier,
        max_requests_per_minute=max_requests_per_minute,
        max_requests_per_day_per_api=max_requests_per_day_per_api,
        minute_quota_scope="account_global"
        if max_requests_per_minute is not None
        else "unknown",
        daily_quota_scope="per_api"
        if max_requests_per_day_per_api is not None
        else "unknown",
        confidence="MEDIUM" if source != "unknown" else "LOW",
        notes=["Tushare quota profile loaded from project settings."],
    )


def normalized_request_hash(
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
) -> str:
    payload = {
        "api_name": api_name,
        "params": _normalized(params),
        "fields": sorted(str(field) for field in fields),
    }
    return _stable_hash(payload)


def new_usage_record(
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
    status: UsageStatus,
    execution_mode: ExecutionMode,
    run_id: str | None = None,
    row_count: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    token_hash: str | None = None,
    planned_at: datetime | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> TushareUsageRecord:
    now = _as_naive_utc(None)
    return TushareUsageRecord(
        request_id=str(uuid4()),
        run_id=run_id,
        api_name=api_name,
        params_hash=normalized_request_hash(
            api_name=api_name,
            params=params,
            fields=fields,
        ),
        params_redacted=redact_params(params),
        fields=fields,
        planned_at=planned_at or now,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        row_count=row_count,
        error_type=error_type,
        error_message=error_message,
        token_hash=token_hash,
        execution_mode=execution_mode,
    )


def token_hash(token: str | None) -> str | None:
    if not token:
        return None
    return hashlib.sha256(token.encode()).hexdigest()


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): ("<redacted>" if "token" in str(key).lower() else value)
        for key, value in params.items()
    }


def _decision(
    status: DecisionStatus,
    reason: str,
    cost: TushareFetchCostEstimate,
    state: TushareQuotaState,
    *,
    safe_to_execute_now: bool,
    recommended_batch_size: int | None = None,
    recommended_batches: list[dict[str, Any]] | None = None,
) -> TushareBudgetDecision:
    return TushareBudgetDecision(
        status=status,
        reason=reason,
        estimated_request_count=cost.estimated_request_count,
        net_new_request_count=cost.net_new_request_count,
        quota_profile=state.profile,
        quota_state=state,
        safe_to_execute_now=safe_to_execute_now,
        recommended_batch_size=recommended_batch_size,
        recommended_batches=recommended_batches or [],
        user_message=(
            f"Tushare plan requires {cost.net_new_request_count} net-new requests. "
            f"Profile={state.profile.tier or state.profile.source}, "
            f"minute_limit={state.profile.max_requests_per_minute}, "
            f"daily_limit_per_api={state.profile.max_requests_per_day_per_api}. "
            f"Decision={status}."
        ),
    )


def _recommended_batches(total: int, first_batch_size: int) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    remaining = total
    index = 1
    next_size = first_batch_size
    while remaining > 0:
        size = min(next_size, remaining)
        batches.append(
            {
                "batch": index,
                "request_count": size,
                "wait_seconds_before_start": 0 if index == 1 else 60,
            }
        )
        remaining -= size
        index += 1
        next_size = first_batch_size
    return batches


def _record_to_row(record: TushareUsageRecord) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    data["params_redacted"] = json.dumps(
        data["params_redacted"],
        ensure_ascii=True,
        sort_keys=True,
    )
    data["fields"] = json.dumps(data["fields"], ensure_ascii=True)
    return data


def _row_to_record(row: dict[str, Any]) -> TushareUsageRecord:
    payload = dict(row)
    for key, value in list(payload.items()):
        if pd.isna(value):
            payload[key] = None
    payload["params_redacted"] = json.loads(str(payload.get("params_redacted") or "{}"))
    payload["fields"] = json.loads(str(payload.get("fields") or "[]"))
    return TushareUsageRecord.model_validate(payload)


def _empty_ledger_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "request_id",
            "run_id",
            "api_name",
            "params_hash",
            "params_redacted",
            "fields",
            "planned_at",
            "started_at",
            "finished_at",
            "status",
            "row_count",
            "error_type",
            "error_message",
            "token_hash",
            "execution_mode",
        ]
    )


def _as_naive_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(tz=UTC)
    if current.tzinfo is not None:
        current = current.astimezone(UTC).replace(tzinfo=None)
    return current


def _datetime_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)


def _normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalized(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple | set):
        return [_normalized(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
