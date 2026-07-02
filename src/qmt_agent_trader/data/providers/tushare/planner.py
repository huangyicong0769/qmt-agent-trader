"""Planner for registry-driven Tushare fetches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from qmt_agent_trader.data.providers.base import FetchItem, FetchPlan
from qmt_agent_trader.data.providers.tushare.registry import (
    EndpointSpec,
    TushareEndpointRegistry,
    default_tushare_registry,
)

TS_CODE_PATTERN = re.compile(r"^[0-9A-Z]{5,8}\.(SH|SZ|BJ|HK|SI)$")


@dataclass(frozen=True)
class TusharePlannerConfig:
    symbol_fanout_threshold: int = 30
    autonomous_request_budget: int = 25
    manual_request_budget: int = 500
    max_days_per_batch: int = 366


class TushareFetchPlanner:
    def __init__(
        self,
        registry: TushareEndpointRegistry | None = None,
        *,
        config: TusharePlannerConfig | None = None,
    ) -> None:
        self.registry = registry or default_tushare_registry()
        self.config = config or TusharePlannerConfig()

    def plan(
        self,
        items: list[FetchItem],
        *,
        requested_by_llm: bool = False,
        storage_mode: str = "persistent",
    ) -> FetchPlan:
        planned_items: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        estimated_request_count = 0
        for item in items:
            item_plan = self._plan_item(item)
            status = str(item_plan.get("status"))
            if status != "planned":
                errors.append(item_plan)
                continue
            planned_items.append(item_plan)
            estimated_request_count += int(item_plan["estimated_request_count"])

        if errors and not planned_items:
            first = errors[0]
            return FetchPlan(
                status=str(first.get("status", "INVALID_REQUEST")),
                source="tushare",
                estimated_request_count=estimated_request_count,
                reason=str(first.get("reason", "validation_failed")),
                message=str(first.get("message", "")) or None,
                errors=errors,
            )
        budget = (
            self.config.autonomous_request_budget
            if requested_by_llm
            else self.config.manual_request_budget
        )
        if estimated_request_count > budget:
            return FetchPlan(
                status="BLOCKED",
                source="tushare",
                items=planned_items,
                estimated_request_count=estimated_request_count,
                reason="REQUEST_BUDGET_EXCEEDED",
                message=f"estimated request count {estimated_request_count} exceeds {budget}",
                errors=errors,
            )
        return FetchPlan(
            status="planned",
            source="tushare",
            items=planned_items,
            estimated_request_count=estimated_request_count,
            errors=errors,
            message=f"storage_mode={storage_mode}",
        )

    def _plan_item(self, item: FetchItem) -> dict[str, Any]:
        spec = self.registry.get(item.api_name)
        if spec is None:
            return {
                "status": "INVALID_REQUEST",
                "api_name": item.api_name,
                "reason": "unknown_endpoint",
                "message": "endpoint is not registered",
            }
        if not spec.implemented:
            return {
                "status": "NOT_IMPLEMENTED",
                "api_name": item.api_name,
                "reason": "endpoint_registered_as_placeholder",
                "message": "endpoint is registered as placeholder and cannot be fetched",
            }
        fields = list(item.fields or spec.default_fields)
        unknown_fields = sorted(set(fields).difference(spec.fields))
        if unknown_fields:
            return {
                "status": "INVALID_REQUEST",
                "api_name": spec.api_name,
                "reason": "unknown_fields",
                "unknown_fields": unknown_fields,
                "allowed_fields": list(spec.fields),
            }
        invalid_symbols = [symbol for symbol in item.symbols if not _valid_ts_code(symbol)]
        if invalid_symbols:
            return {
                "status": "INVALID_REQUEST",
                "api_name": spec.api_name,
                "reason": "invalid_ts_code",
                "invalid_symbols": invalid_symbols,
            }
        if spec.symbol_param and _param_required(spec, spec.symbol_param) and not item.symbols:
            return {
                "status": "INVALID_REQUEST",
                "api_name": spec.api_name,
                "reason": "missing_required_symbol",
                "missing_inputs": [spec.symbol_param],
            }
        missing = [
            name
            for name, meta in spec.params.items()
            if bool(meta.get("required")) and name not in _item_params(item, spec)
        ]
        if missing:
            return {
                "status": "INVALID_REQUEST",
                "api_name": spec.api_name,
                "reason": "missing_required_params",
                "missing_inputs": missing,
            }
        strategy = _strategy_for(spec, item, self.config)
        if strategy == "not_implemented":
            return {
                "status": "NOT_IMPLEMENTED",
                "api_name": spec.api_name,
                "reason": "strategy_not_implemented",
            }
        if strategy == "blocked_too_large":
            return {
                "status": "BLOCKED",
                "api_name": spec.api_name,
                "reason": "REQUEST_BUDGET_EXCEEDED",
                "message": "symbol fanout exceeds planner threshold",
            }
        batches = _batches_for(spec, item, fields, strategy, self.config)
        pagination = spec.pagination
        if pagination.get("type") == "limit_offset":
            batches = _expand_pagination(batches, pagination)
        return {
            "status": "planned",
            "api_name": spec.api_name,
            "dataset_id": spec.dataset_id,
            "category": spec.category,
            "strategy": strategy,
            "fields": fields,
            "symbols": list(item.symbols),
            "params": _item_params(item, spec),
            "batches": batches,
            "estimated_request_count": len(batches),
            "target_dataset": spec.raw_dataset_name,
            "target_view": spec.raw_view_name,
            "target_tables": [f"raw/{spec.raw_dataset_name}"],
            "wide_table_targets": list(spec.wide_table_targets),
            "key_columns": list(spec.key_columns),
            "pagination": spec.pagination,
            "pit": spec.pit,
            "doc_url": spec.doc_url,
        }


def _item_params(item: FetchItem, spec: EndpointSpec) -> dict[str, Any]:
    params = dict(item.params)
    for name in ("trade_date", "start_date", "end_date"):
        value = getattr(item, name)
        if value is not None and name in spec.params:
            params[name] = _normalize_date(value)
    for macro_from, macro_to in (
        ("start_date", "start_m"),
        ("end_date", "end_m"),
        ("start_date", "start_q"),
        ("end_date", "end_q"),
    ):
        value = getattr(item, macro_from)
        if value is not None and macro_to in spec.params and macro_to not in params:
            params[macro_to] = _normalize_macro_period(value, macro_to)
    return params


def _strategy_for(
    spec: EndpointSpec,
    item: FetchItem,
    config: TusharePlannerConfig,
) -> str:
    if not spec.implemented:
        return "not_implemented"
    if spec.category == "macro":
        return "symbolless_range"
    if spec.api_name in {"stock_basic", "fund_basic", "index_basic"} and not item.symbols:
        return "full_refresh"
    if item.symbols and spec.supports_symbol_range:
        return (
            "fanout_by_symbol_range"
            if len(item.symbols) <= config.symbol_fanout_threshold
            else "blocked_too_large"
        )
    if spec.supports_marketwide_by_date and item.trade_date:
        return "marketwide_by_trade_date"
    if spec.supports_marketwide_by_date and item.start_date and item.end_date:
        return "marketwide_by_trade_date"
    if not item.symbols:
        return "full_refresh"
    if spec.symbol_param:
        return "fanout_by_symbol_range"
    return "symbolless_range"


def _batches_for(
    spec: EndpointSpec,
    item: FetchItem,
    fields: list[str],
    strategy: str,
    config: TusharePlannerConfig,
) -> list[dict[str, Any]]:
    base = _item_params(item, spec)
    if strategy == "blocked_too_large":
        return []
    if strategy == "fanout_by_symbol_range" and spec.symbol_param:
        return [
            {
                "api_name": spec.api_name,
                "params": {**base, spec.symbol_param: symbol},
                "fields": fields,
                "dataset_id": spec.dataset_id,
            }
            for symbol in item.symbols
        ]
    if strategy == "marketwide_by_trade_date" and item.start_date and item.end_date:
        return [
            {
                "api_name": spec.api_name,
                "params": {**base, "start_date": start, "end_date": end},
                "fields": fields,
                "dataset_id": spec.dataset_id,
            }
            for start, end in _date_chunks(
                item.start_date,
                item.end_date,
                config.max_days_per_batch,
            )
        ]
    return [
        {
            "api_name": spec.api_name,
            "params": base,
            "fields": fields,
            "dataset_id": spec.dataset_id,
        }
    ]


def _expand_pagination(
    batches: list[dict[str, Any]],
    pagination: dict[str, Any],
) -> list[dict[str, Any]]:
    limit_param = str(pagination.get("limit_param", "limit"))
    offset_param = str(pagination.get("offset_param", "offset"))
    page_size = int(pagination.get("page_size", 5000))
    # Execution continues past the first page only when the page is full. The
    # planner records the first page params and pagination contract.
    expanded: list[dict[str, Any]] = []
    for batch in batches:
        params = dict(batch["params"])
        params.setdefault(limit_param, page_size)
        params.setdefault(offset_param, 0)
        expanded.append({**batch, "params": params, "pagination": pagination})
    return expanded


def _date_chunks(start: str, end: str, max_days: int) -> list[tuple[str, str]]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if start_date > end_date:
        return []
    chunks: list[tuple[str, str]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end_date)
        chunks.append((_format_date(cursor), _format_date(chunk_end)))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _param_required(spec: EndpointSpec, name: str) -> bool:
    return bool(spec.params.get(name, {}).get("required", False))


def _valid_ts_code(value: str) -> bool:
    return bool(TS_CODE_PATTERN.match(value))


def _normalize_date(value: str) -> str:
    return str(value).replace("-", "")


def _normalize_macro_period(value: str, target: str) -> str:
    text = _normalize_date(value)
    if target.endswith("_m"):
        return text[:6]
    if target.endswith("_q"):
        if "Q" in str(value):
            return str(value)
        month = int(text[4:6])
        return f"{text[:4]}Q{((month - 1) // 3) + 1}"
    return text


def _parse_date(value: str) -> date:
    normalized = _normalize_date(value)
    return datetime.strptime(normalized, "%Y%m%d").date()


def _format_date(value: date) -> str:
    return value.strftime("%Y%m%d")
