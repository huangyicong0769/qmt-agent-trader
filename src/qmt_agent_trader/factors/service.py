"""Factor computation service backed by canonical daily bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from qmt_agent_trader.data.contracts import (
    AlignmentPolicy,
    ContractBundle,
    CoverageEvidence,
    FetchShape,
    FetchShapeName,
    PITModel,
    SourceCapability,
    SourceCapabilityMatch,
    TargetCalendar,
    coverage_evidence_from_panel,
    find_capability_match,
    observation_grid_from_panel,
    repair_plan_from_evidence,
    source_capability_from_field_source,
)
from qmt_agent_trader.data.field_sources import (
    FieldSourceIndex,
    FieldSourceSpec,
    FillPolicy,
    fetch_columns_for_source,
)
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.providers.tushare.registry import default_tushare_registry
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.input_panel import build_target_frequency_panel
from qmt_agent_trader.factors.registry import FactorRegistry, input_requirements_for_factor


@dataclass(frozen=True)
class FactorComputeResult:
    name: str
    date: str
    path: str
    rows: int
    non_null: int

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "computed",
            "name": self.name,
            "date": self.date,
            "path": self.path,
            "rows": self.rows,
            "non_null": self.non_null,
        }


@dataclass(frozen=True)
class FactorValidationResult:
    name: str
    start: str
    end: str
    actual_data_start: str
    actual_data_end: str
    data_freshness: str
    observations: int
    non_null: int
    coverage: float
    ic_mean: float | None
    ic_by_date: dict[str, float]
    symbols: tuple[str, ...] = ()
    evaluation_mode: str = "cross_sectional"
    time_series: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "validated",
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "actual_data_start": self.actual_data_start,
            "actual_data_end": self.actual_data_end,
            "data_freshness": self.data_freshness,
            "observations": self.observations,
            "non_null": self.non_null,
            "coverage": self.coverage,
            "ic_mean": self.ic_mean,
            "ic_by_date": self.ic_by_date,
            "symbols": list(self.symbols),
            "evaluation_mode": self.evaluation_mode,
        }
        if self.time_series is not None:
            payload["time_series"] = self.time_series
        return payload


@dataclass(frozen=True)
class FactorWalkForwardSlice:
    start: str
    end: str
    observations: int
    non_null: int
    coverage: float
    mean_ic: float | None
    long_short_spread: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "observations": self.observations,
            "non_null": self.non_null,
            "coverage": self.coverage,
            "mean_ic": self.mean_ic,
            "long_short_spread": self.long_short_spread,
        }


@dataclass(frozen=True)
class FactorWalkForwardResult:
    name: str
    start: str
    end: str
    window_days: int
    step_days: int
    quantile: float
    slices: tuple[FactorWalkForwardSlice, ...]

    def as_dict(self) -> dict[str, object]:
        positive_count = sum(1 for item in self.slices if _positive_slice(item))
        return {
            "status": "validated",
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "window_days": self.window_days,
            "step_days": self.step_days,
            "quantile": self.quantile,
            "slice_count": len(self.slices),
            "positive_slice_ratio": (
                positive_count / len(self.slices) if self.slices else 0.0
            ),
            "walk_forward": [item.as_dict() for item in self.slices],
        }


@dataclass(frozen=True)
class FactorEvaluationBundle:
    validation: FactorValidationResult
    walk_forward: FactorWalkForwardResult
    quantile_returns: dict[str, object]


def compute_factor_frame(
    bars: pd.DataFrame,
    name: str,
    *,
    registry: FactorRegistry | None = None,
) -> pd.DataFrame:
    data = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True).copy()
    factor_registry = registry or FactorRegistry()
    values = pd.to_numeric(factor_registry.compute(name, data), errors="coerce")

    return pd.DataFrame(
        {
            "symbol": data["symbol"],
            "trade_date": data["trade_date"],
            "factor_name": name,
            "factor_value": values,
        }
    )


def compute_factor_to_lake(
    lake: DataLake,
    *,
    name: str,
    date: str,
    registry_root: str | None = None,
) -> FactorComputeResult:
    target_date = pd.to_datetime(date).date()
    registry = FactorRegistry(Path(registry_root)) if registry_root is not None else None
    factor_registry = registry or FactorRegistry()
    saved = factor_registry.get_factor(name)
    if saved is None:
        raise ValueError(f"factor is not saved in registry: {name}")
    bars = _load_factor_input(lake, name=name, date=target_date, registry=factor_registry)
    if bars.empty:
        raise ValueError(
            "no factor input data found; run data fetch and build_data_table first"
        )

    factor_frame = compute_factor_frame(bars, name, registry=factor_registry)
    output = factor_frame[factor_frame["trade_date"] == target_date].reset_index(drop=True)
    if output.empty:
        raise ValueError(f"no factor rows for {target_date}")

    dataset_name = f"factor_{name}_{target_date:%Y%m%d}"
    path = lake.write_parquet(output, "gold", dataset_name)
    lake.register_parquet(dataset_name, "gold", dataset_name)
    return FactorComputeResult(
        name=name,
        date=f"{target_date:%Y%m%d}",
        path=str(path),
        rows=len(output),
        non_null=int(output["factor_value"].notna().sum()),
    )


def _load_factor_input(
    lake: DataLake,
    *,
    name: str,
    date: pd.Timestamp | Any,
    registry: FactorRegistry,
) -> pd.DataFrame:
    saved = registry.get_factor(name)
    if saved is None:
        return pd.DataFrame()
    target_date = _ensure_date(date)
    lookback = int(saved.lookback)
    load_start = target_date - timedelta(days=max(lookback, 0))
    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start=load_start,
        target_end=target_date,
        required_fields=list(saved.required_columns),
    )
    _raise_for_unresolved_factor_inputs(name, metadata)
    return panel


def validate_factor(
    lake: DataLake,
    *,
    name: str,
    start: str,
    end: str,
    registry_root: str | None = None,
    symbols: list[str] | None = None,
) -> FactorValidationResult:
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    registry = FactorRegistry(Path(registry_root)) if registry_root is not None else None
    validation = _factor_validation_frame(
        lake,
        name=name,
        start_date=start_date,
        end_date=end_date,
        registry=registry,
        symbols=symbols,
    )
    return _validation_result_from_frame(name, start_date, end_date, validation)


def walk_forward_factor_validation(
    lake: DataLake,
    *,
    name: str,
    start: str,
    end: str,
    window_days: int = 63,
    step_days: int = 63,
    quantile: float = 0.20,
    registry_root: str | None = None,
    symbols: list[str] | None = None,
) -> FactorWalkForwardResult:
    if window_days <= 1:
        raise ValueError("window_days must be greater than 1")
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if quantile <= 0 or quantile > 0.5:
        raise ValueError("quantile must be in (0, 0.5]")

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    registry = FactorRegistry(Path(registry_root)) if registry_root is not None else None
    validation = _factor_validation_frame(
        lake,
        name=name,
        start_date=start_date,
        end_date=end_date,
        registry=registry,
        symbols=symbols,
    )
    return _walk_forward_result_from_frame(
        name,
        start_date,
        end_date,
        validation,
        window_days=window_days,
        step_days=step_days,
        quantile=quantile,
    )


def evaluate_factor(
    lake: DataLake,
    *,
    name: str,
    start: str,
    end: str,
    registry_root: str | None = None,
    symbols: list[str] | None = None,
    window_days: int = 63,
    step_days: int = 63,
    quantile: float = 0.20,
) -> FactorEvaluationBundle:
    if window_days <= 1:
        raise ValueError("window_days must be greater than 1")
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if quantile <= 0 or quantile > 0.5:
        raise ValueError("quantile must be in (0, 0.5]")

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    registry = FactorRegistry(Path(registry_root)) if registry_root is not None else None
    validation_frame = _factor_validation_frame(
        lake,
        name=name,
        start_date=start_date,
        end_date=end_date,
        registry=registry,
        symbols=symbols,
    )
    validation = _validation_result_from_frame(name, start_date, end_date, validation_frame)
    walk_forward = _walk_forward_result_from_frame(
        name,
        start_date,
        end_date,
        validation_frame,
        window_days=window_days,
        step_days=step_days,
        quantile=quantile,
    )
    spreads = [
        item.long_short_spread
        for item in walk_forward.slices
        if item.long_short_spread is not None
    ]
    return FactorEvaluationBundle(
        validation=validation,
        walk_forward=walk_forward,
        quantile_returns={
            "long_short_spread_mean": sum(spreads) / len(spreads) if spreads else 0,
            "walk_forward_slices": len(walk_forward.slices),
        },
    )


def _factor_validation_frame(
    lake: DataLake,
    *,
    name: str,
    start_date: Any,
    end_date: Any,
    registry: FactorRegistry | None,
    symbols: list[str] | None,
) -> pd.DataFrame:
    factor_registry = registry or FactorRegistry()
    saved = factor_registry.get_factor(name)
    lookback = int(saved.lookback) if saved is not None else 0
    load_start = start_date - timedelta(days=max(lookback, 0))
    load_end = end_date + timedelta(days=1)
    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start=f"{load_start:%Y%m%d}",
        target_end=f"{load_end:%Y%m%d}",
        required_fields=list(saved.required_columns) if saved is not None else [],
        symbols=symbols,
    )
    if panel.empty:
        raise ValueError(
            "no daily bars found in data lake; run data fetch and build_data_table first"
        )
    _raise_for_unresolved_factor_inputs(name, metadata)
    factor_frame = compute_factor_frame(panel, name, registry=factor_registry)
    validation = factor_frame.merge(
        _forward_returns(panel),
        on=["symbol", "trade_date"],
        how="inner",
    )
    validation = validation[
        (validation["trade_date"] >= start_date) & (validation["trade_date"] <= end_date)
    ].reset_index(drop=True)
    if validation.empty:
        raise ValueError(f"no validation rows between {start_date} and {end_date}")
    return validation


def check_factor_input_readiness(
    lake: DataLake,
    *,
    factor_name: str,
    start: str,
    end: str,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    factor_registry = FactorRegistry()
    saved = factor_registry.get_factor(factor_name)
    if saved is None:
        return {
            "status": "INVALID_REQUEST",
            "target_frequency": Frequency.DAILY.value,
            "factor_name": factor_name,
            "required_columns": [],
            "field_sources": {},
            "fill_policy_by_field": {},
            "coverage_by_field": {},
            "missing_fields": {},
            "unresolved_fields": [
                {
                    "field": factor_name,
                    "reason": "factor_not_found",
                }
            ],
            "repair_action": {
                "type": "fix_request_argument",
                "reason": "factor_not_found",
            },
            "verification_action": {},
        }
    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start=start,
        target_end=end,
        required_fields=list(saved.required_columns),
        symbols=symbols,
    )
    contract_bundle = _factor_input_contract_bundle(
        panel,
        saved=saved,
        metadata=metadata,
    )
    repair_action = _factor_input_repair_action(
        metadata,
        symbols=symbols or [],
        start=start,
        end=end,
        contract_bundle=contract_bundle,
    )
    status = _readiness_status(panel, metadata, saved.required_columns)
    if status == "OK" and _contract_has_blockers(contract_bundle):
        status = "PARTIAL_COVERAGE"
    return {
        "status": status,
        "contract_status": _contract_readiness_status(contract_bundle),
        "reason": (
            "UNSATISFIED_DATA_REQUIREMENT"
            if _contract_has_blockers(contract_bundle)
            else "DATA_REQUIREMENTS_SATISFIED"
        ),
        "target_frequency": Frequency.DAILY.value,
        "factor_name": factor_name,
        "required_columns": list(saved.required_columns),
        "requirements": [
            requirement.model_dump(mode="json")
            for requirement in contract_bundle.requirements
        ],
        "source_matches": [
            match.model_dump(mode="json")
            for match in contract_bundle.source_matches
        ],
        "observation_grid": (
            contract_bundle.observation_grid.model_dump(mode="json")
            if contract_bundle.observation_grid is not None
            else None
        ),
        "coverage_evidence": [
            evidence.model_dump(mode="json")
            for evidence in contract_bundle.coverage_evidence
        ],
        "repair_plans": [
            plan.model_dump(mode="json")
            for plan in contract_bundle.repair_plans
        ],
        "field_sources": metadata["field_sources"],
        "fill_policy_by_field": {
            field: source["fill_policy"]
            for field, source in metadata["field_sources"].items()
        },
        "coverage_by_field": metadata["coverage_by_field"],
        "missing_fields": metadata["missing_fields"],
        "unresolved_fields": metadata["unresolved_fields"],
        "repair_action": repair_action,
        "verification_action": {
            "function": "check_factor_input_readiness",
            "input": {
                "factor_name": factor_name,
                "start": start,
                "end": end,
                "symbols": symbols or [],
            },
        },
    }


def _raise_for_unresolved_factor_inputs(name: str, metadata: dict[str, Any]) -> None:
    unresolved = metadata.get("unresolved_fields") or []
    if not unresolved:
        return
    raise ValueError(
        f"factor '{name}' has unresolved input fields: {unresolved}; "
        f"repair_action={_factor_input_repair_action(metadata, symbols=[], start=None, end=None)}"
    )


def _readiness_status(
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    required_columns: tuple[str, ...],
) -> str:
    if metadata.get("status") == "NO_DATA" or panel.empty:
        return "NO_DATA"
    if metadata.get("unresolved_fields"):
        return "INVALID_REQUEST"
    coverage = metadata.get("coverage_by_field", {})
    data_fields = [
        field for field in required_columns if field not in {"symbol", "trade_date"}
    ]
    if metadata.get("missing_fields"):
        return "PARTIAL_COVERAGE"
    if any((coverage.get(field, {}).get("non_null") or 0) == 0 for field in data_fields):
        return "PARTIAL_COVERAGE"
    return "OK"


def _factor_input_repair_action(
    metadata: dict[str, Any],
    *,
    symbols: list[str],
    start: str | None,
    end: str | None,
    contract_bundle: ContractBundle | None = None,
) -> dict[str, Any]:
    if contract_bundle is not None and contract_bundle.repair_plans:
        return _contract_repair_action(
            contract_bundle,
            symbols=symbols,
            start=start,
            end=end,
        )
    unresolved = metadata.get("unresolved_fields") or []
    if unresolved:
        event_fields = [
            item
            for item in unresolved
            if item.get("reason") == "event_field_requires_explicit_transform"
        ]
        if event_fields:
            return {
                "type": "event_transform_required",
                "tool": None,
                "reason": "event_field_requires_explicit_transform",
                "fields": [str(item["field"]) for item in event_fields],
                "candidates": [
                    {"field": str(item["field"]), "api_name": item.get("api_name")}
                    for item in event_fields
                ],
            }
        ambiguous = [item for item in unresolved if item.get("status") == "AMBIGUOUS_FIELD_SOURCE"]
        if ambiguous:
            first = ambiguous[0]
            return {
                "type": "AMBIGUOUS_FIELD_SOURCE",
                "tool": "list_tushare_capabilities",
                "field": first.get("field"),
                "candidates": first.get("candidates", []),
            }
        return {
            "type": "capability_discovery_required",
            "tool": "list_tushare_capabilities",
            "reason": "unknown_field_source",
            "fields": [str(item.get("field")) for item in unresolved],
        }

    missing_fields = metadata.get("missing_fields") or {}
    if not missing_fields:
        return {}
    source_index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())
    grouped: dict[str, tuple[FieldSourceSpec, list[str]]] = {}
    for field in sorted(missing_fields):
        source = source_index.best_source_for_field(field, target_frequency=Frequency.DAILY)
        if source is None:
            continue
        _source, fields = grouped.setdefault(source.api_name, (source, []))
        fields.append(field)
    fetch_items = [
        _fetch_item_for_source(
            source,
            fields=fields,
            symbols=symbols,
            start=start,
            end=end,
        )
        for source, fields in grouped.values()
    ]
    return {
        "type": "fetch_missing_data",
        "tool": "run_tushare_fetch",
        "reason": _repair_reason([source for source, _fields in grouped.values()]),
        "fetch_items": fetch_items,
        "execute_plan": True,
    }


def _factor_input_contract_bundle(
    panel: pd.DataFrame,
    *,
    saved: Any,
    metadata: dict[str, Any],
) -> ContractBundle:
    requirements = list(input_requirements_for_factor(saved))
    grid = observation_grid_from_panel(
        panel,
        target_frequency=Frequency.DAILY,
        calendar=TargetCalendar.TRADING_DAYS,
    )
    registry = default_tushare_registry()
    source_index = FieldSourceIndex.from_tushare_registry(registry)
    matches: list[SourceCapabilityMatch] = []
    evidence_items: list[CoverageEvidence] = []
    repair_plans = []
    for requirement in requirements:
        match = _match_requirement_to_source(
            requirement,
            source_index=source_index,
            metadata=metadata,
        )
        matched_source = match.source if match is not None else None
        evidence = coverage_evidence_from_panel(
            panel,
            requirement=requirement,
            matched_source=matched_source,
            grid=grid,
        )
        evidence_items.append(evidence)
        if match is not None:
            matches.append(match)
            repair_plan = repair_plan_from_evidence(
                requirement=requirement,
                match=match,
                evidence=evidence,
                grid=grid,
            )
            if repair_plan is not None:
                repair_plans.append(repair_plan)
    return ContractBundle(
        requirements=requirements,
        source_matches=matches,
        observation_grid=grid,
        coverage_evidence=evidence_items,
        repair_plans=repair_plans,
    )


def _match_requirement_to_source(
    requirement: Any,
    *,
    source_index: FieldSourceIndex,
    metadata: dict[str, Any],
) -> SourceCapabilityMatch | None:
    if requirement.field in metadata.get("field_sources", {}):
        field_source = source_index.best_source_for_field(
            requirement.field,
            target_frequency=requirement.target_frequency,
        )
        if field_source is None:
            return None
        endpoint = default_tushare_registry().require(field_source.api_name)
        capability = source_capability_from_field_source(field_source, endpoint)
        return find_capability_match(requirement, [capability])
    if requirement.field in {"open", "high", "low", "close", "vol", "amount", "turnover"}:
        return SourceCapabilityMatch(
            requirement=requirement,
            source=_canonical_daily_bars_capability(requirement.field),
            selected_fetch_shape=FetchShape(
                name=FetchShapeName.SYMBOL_TIME_RANGE,
                unit="canonical daily bars over a date range",
                symbol_param="ts_code",
                start_param="start_date",
                end_param="end_date",
            ),
            explanation="field is already provided by canonical daily bar skeleton",
        )
    candidates = []
    for source in source_index.sources_for_field(requirement.field):
        endpoint = default_tushare_registry().require(source.api_name)
        candidates.append(source_capability_from_field_source(source, endpoint))
    return find_capability_match(requirement, candidates)


def _canonical_daily_bars_capability(field: str) -> SourceCapability:
    return SourceCapability(
        source_id="canonical.daily_bars",
        provider="local",
        api_name="daily_bars",
        raw_dataset_name="tushare/daily",
        canonical_dataset_name="daily_bars",
        fields=(
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
            "turnover",
        ),
        native_frequency=Frequency.DAILY,
        time_axis="trade_date",
        entity_axis="symbol",
        fetch_shapes=(
            FetchShape(
                name=FetchShapeName.SYMBOL_TIME_RANGE,
                unit="canonical daily bars over a date range",
                symbol_param="ts_code",
                start_param="start_date",
                end_param="end_date",
            ),
        ),
        pit_model=PITModel(
            visible_time_field="trade_date",
            visible_time_semantics="same-day market data visibility",
        ),
        default_alignment_policy=AlignmentPolicy.EXACT,
        limitations=(),
    )


def _contract_repair_action(
    contract_bundle: ContractBundle,
    *,
    symbols: list[str],
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    fetch_items = []
    for plan in contract_bundle.repair_plans:
        fields = list(plan.fetch_plan.fields)
        source_index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())
        field_source = source_index.best_source_for_field(
            plan.requirement.field,
            target_frequency=plan.requirement.target_frequency,
        )
        if field_source is None:
            continue
        fetch_items.append(
            _fetch_item_for_source(
                field_source,
                fields=fields,
                symbols=symbols,
                start=start,
                end=end,
            )
        )
    return {
        "type": "fetch_missing_data",
        "tool": "run_tushare_fetch",
        "reason": "UNSATISFIED_OBSERVATION_COVERAGE",
        "fetch_items": fetch_items,
        "items": fetch_items,
        "execute_plan": True,
        "contract_repair_plans": [
            plan.model_dump(mode="json")
            for plan in contract_bundle.repair_plans
        ],
        "verification_actions": [
            "build_data_table(daily_market)",
            "check_factor_input_readiness",
            "run_backtest",
        ],
    }


def _contract_readiness_status(contract_bundle: ContractBundle) -> str:
    if not contract_bundle.requirements:
        return "INVALID_REQUIREMENT"
    if any(
        evidence.status == "UNRESOLVED"
        for evidence in contract_bundle.coverage_evidence
    ):
        return "UNRESOLVED_SOURCE"
    if _contract_has_blockers(contract_bundle):
        return "PARTIAL_REPAIRABLE" if contract_bundle.repair_plans else "PARTIAL_INFEASIBLE"
    return "READY"


def _contract_has_blockers(contract_bundle: ContractBundle) -> bool:
    return any(
        evidence.status != "SATISFIED"
        for evidence in contract_bundle.coverage_evidence
    )


def _fetch_item_for_source(
    source: FieldSourceSpec,
    *,
    fields: list[str],
    symbols: list[str],
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "api_name": source.api_name,
        "symbols": symbols,
        "fields": fetch_columns_for_source(source, sorted(fields)),
    }
    if start is not None:
        item["start_date"] = _fetch_start_for_source(source, start)
    if end is not None:
        item["end_date"] = end
    return item


def _fetch_start_for_source(source: FieldSourceSpec, start: str) -> str:
    if source.fill_policy is FillPolicy.EXACT:
        return start
    parsed = pd.to_datetime(start).date()
    return f"{parsed - pd.DateOffset(years=1):%Y%m%d}"


def _repair_reason(sources: list[FieldSourceSpec]) -> str:
    if sources and all(source.fill_policy is FillPolicy.EXACT for source in sources):
        return "missing_exact_daily_coverage"
    if sources and all(source.fill_policy is FillPolicy.ASOF_SNAPSHOT for source in sources):
        return "missing_pit_snapshot_coverage"
    return "missing_factor_input_coverage"


def _ensure_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(text).date()


def _validation_result_from_frame(
    name: str,
    start_date: Any,
    end_date: Any,
    validation: pd.DataFrame,
) -> FactorValidationResult:
    actual_start = min(validation["trade_date"])
    actual_end = max(validation["trade_date"])
    data_freshness = (
        "stale_vs_requested_end"
        if actual_end < end_date
        else "covers_requested_end"
    )

    usable = validation.dropna(subset=["factor_value", "forward_return_1d"])
    evaluated_symbols = tuple(sorted(validation["symbol"].astype(str).unique()))
    ic_by_date: dict[str, float] = {}
    for trade_date, group in usable.groupby("trade_date"):
        if len(group) < 2:
            continue
        ic = group["factor_value"].corr(group["forward_return_1d"], method="spearman")
        if pd.notna(ic):
            ic_by_date[f"{trade_date:%Y%m%d}"] = float(ic)

    ic_mean = float(pd.Series(ic_by_date).mean()) if ic_by_date else None
    time_series = (
        _time_series_factor_metrics(usable)
        if len(evaluated_symbols) == 1
        else None
    )
    observations = len(validation)
    non_null = len(usable)
    return FactorValidationResult(
        name=name,
        start=f"{start_date:%Y%m%d}",
        end=f"{end_date:%Y%m%d}",
        actual_data_start=f"{actual_start:%Y%m%d}",
        actual_data_end=f"{actual_end:%Y%m%d}",
        data_freshness=data_freshness,
        observations=observations,
        non_null=non_null,
        coverage=non_null / observations if observations else 0.0,
        ic_mean=ic_mean,
        ic_by_date=ic_by_date,
        symbols=evaluated_symbols,
        evaluation_mode="time_series" if time_series is not None else "cross_sectional",
        time_series=time_series,
    )


def _walk_forward_result_from_frame(
    name: str,
    start_date: Any,
    end_date: Any,
    validation: pd.DataFrame,
    *,
    window_days: int,
    step_days: int,
    quantile: float,
) -> FactorWalkForwardResult:
    dates = sorted(validation["trade_date"].unique())
    slices: list[FactorWalkForwardSlice] = []
    for index in range(0, len(dates), step_days):
        window_dates = dates[index : index + window_days]
        if len(window_dates) < 2:
            continue
        frame = validation[validation["trade_date"].isin(window_dates)]
        slices.append(_walk_forward_slice(frame, quantile=quantile))

    return FactorWalkForwardResult(
        name=name,
        start=f"{start_date:%Y%m%d}",
        end=f"{end_date:%Y%m%d}",
        window_days=window_days,
        step_days=step_days,
        quantile=quantile,
        slices=tuple(slices),
    )


def _forward_returns(bars: pd.DataFrame) -> pd.DataFrame:
    data = bars.sort_values(["symbol", "trade_date"]).copy()
    next_close = data.groupby("symbol")["close"].shift(-1)
    data["forward_return_1d"] = next_close / data["close"] - 1
    return data[["symbol", "trade_date", "forward_return_1d"]]


def _walk_forward_slice(frame: pd.DataFrame, *, quantile: float) -> FactorWalkForwardSlice:
    usable = frame.dropna(subset=["factor_value", "forward_return_1d"])
    ic_values: list[float] = []
    spread_values: list[float] = []
    for _, group in usable.groupby("trade_date"):
        if len(group) < 2:
            continue
        ic = group["factor_value"].corr(group["forward_return_1d"], method="spearman")
        if pd.notna(ic):
            ic_values.append(float(ic))
        ordered = group.sort_values("factor_value")
        bucket_size = max(1, int(len(ordered) * quantile))
        low = ordered.head(bucket_size)["forward_return_1d"].mean()
        high = ordered.tail(bucket_size)["forward_return_1d"].mean()
        if pd.notna(low) and pd.notna(high):
            spread_values.append(float(high - low))

    start = min(frame["trade_date"])
    end = max(frame["trade_date"])
    return FactorWalkForwardSlice(
        start=f"{start:%Y%m%d}",
        end=f"{end:%Y%m%d}",
        observations=len(frame),
        non_null=len(usable),
        coverage=len(usable) / len(frame) if len(frame) else 0.0,
        mean_ic=float(mean(ic_values)) if ic_values else None,
        long_short_spread=float(mean(spread_values)) if spread_values else None,
    )


def _time_series_factor_metrics(usable: pd.DataFrame) -> dict[str, Any]:
    if usable.empty:
        return {
            "observations": 0,
            "pearson_ic": None,
            "spearman_ic": None,
            "direction_hit_rate": None,
            "mean_forward_return": None,
        }
    factor = usable["factor_value"]
    forward = usable["forward_return_1d"]
    pearson = factor.corr(forward, method="pearson") if len(usable) >= 2 else None
    spearman = factor.corr(forward, method="spearman") if len(usable) >= 2 else None
    directional = usable[(factor != 0) & (forward != 0)]
    if directional.empty:
        hit_rate = None
    else:
        directional_hits = directional["factor_value"] * directional["forward_return_1d"] > 0
        hit_rate = float(directional_hits.mean())
    return {
        "observations": len(usable),
        "pearson_ic": _optional_float(pearson),
        "spearman_ic": _optional_float(spearman),
        "direction_hit_rate": hit_rate,
        "mean_forward_return": float(forward.mean()) if pd.notna(forward.mean()) else None,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or not pd.notna(value):
        return None
    return float(value)


def _positive_slice(item: FactorWalkForwardSlice) -> bool:
    return (
        item.mean_ic is not None
        and item.long_short_spread is not None
        and item.mean_ic > 0
        and item.long_short_spread > 0
    )
