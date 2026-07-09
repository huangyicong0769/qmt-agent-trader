"""Contract models for factor data requirements and source capabilities."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from qmt_agent_trader.data.field_sources import FieldSourceSpec, FillPolicy
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.providers.tushare.registry import EndpointSpec


class TargetCalendar(StrEnum):
    TRADING_DAYS = "trading_days"
    CALENDAR_DAYS = "calendar_days"
    REPORT_PERIODS = "report_periods"
    EVENT_TIMES = "event_times"


class EntityScope(StrEnum):
    STOCK_CROSS_SECTION = "stock_cross_section"
    ETF_CROSS_SECTION = "etf_cross_section"
    STOCK_ETF_CROSS_SECTION = "stock_etf_cross_section"
    SINGLE_SYMBOL_SERIES = "single_symbol_series"
    MARKETWIDE_SERIES = "marketwide_series"
    MACRO_SERIES = "macro_series"


class AlignmentPolicy(StrEnum):
    EXACT = "exact"
    ASOF = "asof"
    WINDOW_AGGREGATE = "window_aggregate"
    EVENT_TRANSFORM = "event_transform"
    NO_DEFAULT_FILL = "no_default_fill"


class FetchShapeName(StrEnum):
    MARKETWIDE_TIME_SLICE = "marketwide_time_slice"
    SYMBOL_TIME_RANGE = "symbol_time_range"
    REPORT_PERIOD_SLICE = "report_period_slice"
    EVENT_STREAM_RANGE = "event_stream_range"
    MACRO_TIME_RANGE = "macro_time_range"
    STATIC_FULL_SNAPSHOT = "static_full_snapshot"


class CoverageStatus(StrEnum):
    SATISFIED = "SATISFIED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    UNRESOLVED = "UNRESOLVED"
    INFEASIBLE = "INFEASIBLE"


class ReadinessStatus(StrEnum):
    READY = "READY"
    PARTIAL_REPAIRABLE = "PARTIAL_REPAIRABLE"
    PARTIAL_INFEASIBLE = "PARTIAL_INFEASIBLE"
    UNRESOLVED_SOURCE = "UNRESOLVED_SOURCE"
    BLOCKED_BY_BUDGET = "BLOCKED_BY_BUDGET"
    INVALID_REQUIREMENT = "INVALID_REQUIREMENT"


class CoveragePolicy(BaseModel):
    min_required_field_coverage: float = 0.80
    min_cross_sectional_coverage: float = 0.50
    min_time_coverage: float | None = None


class StalenessPolicy(BaseModel):
    max_staleness_days: int
    p95_staleness_days: int | None = None


class FactorInputRequirement(BaseModel):
    field: str
    target_frequency: Frequency = Frequency.DAILY
    target_calendar: TargetCalendar = TargetCalendar.TRADING_DAYS
    entity_scope: EntityScope = EntityScope.STOCK_CROSS_SECTION
    alignment_policy: AlignmentPolicy = AlignmentPolicy.EXACT
    pit_required: bool = True
    coverage_policy: CoveragePolicy = Field(default_factory=CoveragePolicy)
    staleness_policy: StalenessPolicy | None = None
    allowed_source_frequencies: tuple[Frequency, ...] | None = None

    @property
    def requirement_id(self) -> str:
        return (
            f"{self.field}:{self.target_frequency.value}:"
            f"{self.target_calendar.value}:{self.entity_scope.value}:"
            f"{self.alignment_policy.value}"
        )


class PITModel(BaseModel):
    visible_time_field: str | None = None
    period_field: str | None = None
    visible_time_semantics: str


class FetchShape(BaseModel):
    name: FetchShapeName
    unit: str
    time_param: str | None = None
    start_param: str | None = None
    end_param: str | None = None
    symbol_param: str | None = None
    requires_calendar_expansion: bool = False
    max_rows_per_request: int | None = None


class SourceCapability(BaseModel):
    source_id: str
    provider: str
    api_name: str
    raw_dataset_name: str
    canonical_dataset_name: str | None = None
    fields: tuple[str, ...]
    native_frequency: Frequency
    time_axis: str
    entity_axis: str
    fetch_shapes: tuple[FetchShape, ...]
    pit_model: PITModel
    default_alignment_policy: AlignmentPolicy
    limitations: tuple[str, ...] = ()


class SourceCapabilityMatch(BaseModel):
    requirement: FactorInputRequirement
    source: SourceCapability
    selected_fetch_shape: FetchShape
    status: str = "MATCHED"
    explanation: str


class ObservationGrid(BaseModel):
    time_points: list[str]
    entity_ids: list[str]
    target_frequency: Frequency
    calendar: TargetCalendar
    total_cells: int


class CoverageObligation(BaseModel):
    required_cells: int
    required_time_points: int
    required_entities: int
    min_required_field_coverage: float
    min_cross_sectional_coverage: float
    min_time_coverage: float | None = None
    alignment_policy: AlignmentPolicy
    staleness_policy: StalenessPolicy | None = None


class CoverageEvidence(BaseModel):
    requirement_id: str
    field: str
    matched_source_id: str | None = None
    required_cells: int
    observed_non_null_cells: int
    field_coverage: float
    required_time_points: int
    covered_time_points: int
    time_coverage: float
    required_entities: int
    covered_entities: int
    entity_coverage: float
    cross_sectional_coverage_by_date: dict[str, float]
    cross_sectional_coverage_summary: dict[str, float]
    alignment_policy: AlignmentPolicy
    pit_safe: bool
    staleness_summary: dict[str, float] | None = None
    status: CoverageStatus
    blocking_reasons: list[str]


class FetchPlan(BaseModel):
    tool: str = "run_tushare_fetch"
    source_id: str
    api_name: str
    fetch_shape: FetchShapeName
    expansion: str | None = None
    required_time_points: list[str]
    missing_time_points: list[str]
    required_entities: list[str]
    fields: list[str]
    estimated_requests: int
    estimated_rows: int | None = None


class PlanFeasibility(BaseModel):
    status: str
    blocking_reasons: list[str] = Field(default_factory=list)


class RepairPlan(BaseModel):
    repair_reason: str
    requirement: FactorInputRequirement
    matched_source: SourceCapability
    coverage_gap: CoverageEvidence
    fetch_plan: FetchPlan
    feasibility: PlanFeasibility
    then_run: list[str]


class ObservationPlan(BaseModel):
    requirement: FactorInputRequirement
    matched_source: SourceCapability
    selected_fetch_shape: FetchShape
    observation_grid: ObservationGrid
    coverage_obligation: CoverageObligation
    fetch_plan: FetchPlan
    feasibility: PlanFeasibility
    explanation: str


class ContractBundle(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    requirements: list[FactorInputRequirement]
    source_matches: list[SourceCapabilityMatch]
    observation_grid: ObservationGrid | None
    coverage_evidence: list[CoverageEvidence]
    repair_plans: list[RepairPlan]


def source_capability_from_field_source(
    source: FieldSourceSpec,
    endpoint: EndpointSpec,
) -> SourceCapability:
    fetch_shapes: list[FetchShape] = []
    rows_per_request = endpoint.call_limit.get("rows_per_request")
    max_rows = int(rows_per_request) if isinstance(rows_per_request, int | float) else None
    if endpoint.supports_marketwide_by_date:
        time_param = _point_time_param(endpoint)
        fetch_shapes.append(
            FetchShape(
                name=FetchShapeName.MARKETWIDE_TIME_SLICE,
                unit="one time point returns a market cross-section",
                time_param=time_param,
                requires_calendar_expansion=True,
                max_rows_per_request=max_rows,
            )
        )
    if endpoint.supports_symbol_range:
        fetch_shapes.append(
            FetchShape(
                name=FetchShapeName.SYMBOL_TIME_RANGE,
                unit="one symbol over a date range",
                symbol_param=endpoint.symbol_param,
                start_param=_range_start_param(endpoint),
                end_param=_range_end_param(endpoint),
            )
        )
    if not fetch_shapes:
        fetch_shapes.append(
            FetchShape(
                name=FetchShapeName.STATIC_FULL_SNAPSHOT,
                unit="one full snapshot",
            )
        )

    alignment = _alignment_policy_for_fill(source.fill_policy)
    canonical_dataset = endpoint.wide_table_targets[0] if endpoint.wide_table_targets else None
    return SourceCapability(
        source_id=f"tushare.{source.api_name}",
        provider="tushare",
        api_name=source.api_name,
        raw_dataset_name=source.raw_dataset_name,
        canonical_dataset_name=canonical_dataset,
        fields=endpoint.fields,
        native_frequency=source.frequency,
        time_axis=source.source_time_column or source.visible_time_column or "none",
        entity_axis=source.entity_column or "none",
        fetch_shapes=tuple(fetch_shapes),
        pit_model=PITModel(
            visible_time_field=source.visible_time_column,
            period_field="end_date" if "end_date" in endpoint.fields else None,
            visible_time_semantics=_visible_time_semantics(source),
        ),
        default_alignment_policy=alignment,
        limitations=tuple(_capability_limitations(source, endpoint)),
    )


def find_capability_match(
    requirement: FactorInputRequirement,
    candidates: list[SourceCapability],
) -> SourceCapabilityMatch | None:
    ranked: list[tuple[int, SourceCapability, FetchShape, str]] = []
    for candidate in candidates:
        shape = _select_fetch_shape(requirement, candidate)
        if shape is None:
            continue
        if requirement.field not in candidate.fields:
            continue
        if requirement.pit_required and not candidate.pit_model.visible_time_field:
            continue
        if requirement.allowed_source_frequencies and (
            candidate.native_frequency not in requirement.allowed_source_frequencies
        ):
            continue
        if requirement.alignment_policy != candidate.default_alignment_policy:
            continue
        score = 0
        if candidate.native_frequency == requirement.target_frequency:
            score += 10
        if shape.name is FetchShapeName.MARKETWIDE_TIME_SLICE:
            score += 5
        if shape.name is FetchShapeName.SYMBOL_TIME_RANGE:
            score += 3
        ranked.append(
            (
                score,
                candidate,
                shape,
                (
                    "source satisfies field, frequency, PIT visibility, alignment "
                    "policy, and fetch-shape requirements"
                ),
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    _score, source, shape, explanation = ranked[0]
    return SourceCapabilityMatch(
        requirement=requirement,
        source=source,
        selected_fetch_shape=shape,
        explanation=explanation,
    )


def observation_grid_from_panel(
    panel: pd.DataFrame,
    *,
    target_frequency: Frequency,
    calendar: TargetCalendar,
) -> ObservationGrid:
    if panel.empty:
        time_points: list[str] = []
        entities: list[str] = []
    else:
        time_points = sorted({_format_date(value) for value in panel["trade_date"]})
        entities = sorted({str(value) for value in panel["symbol"]})
    return ObservationGrid(
        time_points=time_points,
        entity_ids=entities,
        target_frequency=target_frequency,
        calendar=calendar,
        total_cells=len(time_points) * len(entities),
    )


def coverage_obligation_for_requirement(
    requirement: FactorInputRequirement,
    grid: ObservationGrid,
) -> CoverageObligation:
    return CoverageObligation(
        required_cells=grid.total_cells,
        required_time_points=len(grid.time_points),
        required_entities=len(grid.entity_ids),
        min_required_field_coverage=requirement.coverage_policy.min_required_field_coverage,
        min_cross_sectional_coverage=requirement.coverage_policy.min_cross_sectional_coverage,
        min_time_coverage=requirement.coverage_policy.min_time_coverage,
        alignment_policy=requirement.alignment_policy,
        staleness_policy=requirement.staleness_policy,
    )


def coverage_evidence_from_panel(
    panel: pd.DataFrame,
    *,
    requirement: FactorInputRequirement,
    matched_source: SourceCapability | None,
    grid: ObservationGrid,
) -> CoverageEvidence:
    if requirement.field in panel.columns:
        observed = panel[["symbol", "trade_date", requirement.field]].copy()
        observed["_has_value"] = observed[requirement.field].notna()
    else:
        observed = panel[["symbol", "trade_date"]].copy() if not panel.empty else pd.DataFrame()
        observed["_has_value"] = False

    non_null = int(observed["_has_value"].sum()) if not observed.empty else 0
    by_date: dict[str, float] = {}
    covered_dates = 0
    for day, group in observed.groupby("trade_date", sort=True):
        coverage = float(group["_has_value"].sum() / len(group)) if len(group) else 0.0
        key = _format_date(day)
        by_date[key] = coverage
        if coverage >= requirement.coverage_policy.min_cross_sectional_coverage:
            covered_dates += 1
    covered_entities = (
        int(observed.loc[observed["_has_value"], "symbol"].astype(str).nunique())
        if not observed.empty
        else 0
    )
    required_cells = grid.total_cells
    required_time_points = len(grid.time_points)
    required_entities = len(grid.entity_ids)
    field_coverage = non_null / required_cells if required_cells else 0.0
    time_coverage = covered_dates / required_time_points if required_time_points else 0.0
    entity_coverage = covered_entities / required_entities if required_entities else 0.0
    summary = _coverage_summary(by_date)
    blocking = _coverage_blockers(
        requirement,
        field_coverage=field_coverage,
        time_coverage=time_coverage,
        entity_coverage=entity_coverage,
        matched_source=matched_source,
    )
    if matched_source is None:
        status = CoverageStatus.UNRESOLVED
    elif non_null == 0:
        status = CoverageStatus.MISSING
    elif blocking:
        status = CoverageStatus.PARTIAL
    else:
        status = CoverageStatus.SATISFIED
    return CoverageEvidence(
        requirement_id=requirement.requirement_id,
        field=requirement.field,
        matched_source_id=matched_source.source_id if matched_source else None,
        required_cells=required_cells,
        observed_non_null_cells=non_null,
        field_coverage=field_coverage,
        required_time_points=required_time_points,
        covered_time_points=covered_dates,
        time_coverage=time_coverage,
        required_entities=required_entities,
        covered_entities=covered_entities,
        entity_coverage=entity_coverage,
        cross_sectional_coverage_by_date=by_date,
        cross_sectional_coverage_summary=summary,
        alignment_policy=requirement.alignment_policy,
        pit_safe=bool(matched_source and matched_source.pit_model.visible_time_field),
        status=status,
        blocking_reasons=blocking,
    )


def repair_plan_from_evidence(
    *,
    requirement: FactorInputRequirement,
    match: SourceCapabilityMatch,
    evidence: CoverageEvidence,
    grid: ObservationGrid,
) -> RepairPlan | None:
    if evidence.status is CoverageStatus.SATISFIED:
        return None
    missing_time_points = [
        day
        for day in grid.time_points
        if evidence.cross_sectional_coverage_by_date.get(day, 0.0)
        < requirement.coverage_policy.min_cross_sectional_coverage
    ]
    fetch_shape = match.selected_fetch_shape
    estimated_requests = _estimated_requests(fetch_shape, missing_time_points, grid.entity_ids)
    max_rows = fetch_shape.max_rows_per_request
    estimated_rows = (
        len(missing_time_points) * len(grid.entity_ids)
        if fetch_shape.name is FetchShapeName.MARKETWIDE_TIME_SLICE
        else None
    )
    if max_rows is not None and fetch_shape.name is FetchShapeName.MARKETWIDE_TIME_SLICE:
        estimated_rows = min(len(grid.entity_ids), max_rows) * len(missing_time_points)
    feasibility = PlanFeasibility(
        status="FEASIBLE" if missing_time_points else "NO_FETCH_UNITS",
        blocking_reasons=[] if missing_time_points else ["no_missing_fetch_units_derived"],
    )
    fetch_plan = FetchPlan(
        source_id=match.source.source_id,
        api_name=match.source.api_name,
        fetch_shape=fetch_shape.name,
        expansion="trade_calendar"
        if fetch_shape.requires_calendar_expansion
        else None,
        required_time_points=grid.time_points,
        missing_time_points=missing_time_points,
        required_entities=grid.entity_ids,
        fields=[requirement.field],
        estimated_requests=estimated_requests,
        estimated_rows=estimated_rows,
    )
    return RepairPlan(
        repair_reason="UNSATISFIED_OBSERVATION_COVERAGE",
        requirement=requirement,
        matched_source=match.source,
        coverage_gap=evidence,
        fetch_plan=fetch_plan,
        feasibility=feasibility,
        then_run=[
            "build_data_table(daily_market)",
            "check_factor_input_readiness",
            "run_backtest",
        ],
    )


def _select_fetch_shape(
    requirement: FactorInputRequirement,
    capability: SourceCapability,
) -> FetchShape | None:
    if (
        requirement.target_calendar is TargetCalendar.TRADING_DAYS
        and requirement.entity_scope
        in {EntityScope.STOCK_CROSS_SECTION, EntityScope.STOCK_ETF_CROSS_SECTION}
    ):
        for shape in capability.fetch_shapes:
            if shape.name is FetchShapeName.MARKETWIDE_TIME_SLICE:
                return shape
    for shape in capability.fetch_shapes:
        if shape.name is FetchShapeName.SYMBOL_TIME_RANGE:
            return shape
    return capability.fetch_shapes[0] if capability.fetch_shapes else None


def _alignment_policy_for_fill(fill_policy: FillPolicy) -> AlignmentPolicy:
    if fill_policy is FillPolicy.EXACT:
        return AlignmentPolicy.EXACT
    if fill_policy is FillPolicy.ASOF_SNAPSHOT:
        return AlignmentPolicy.ASOF
    if fill_policy is FillPolicy.EVENT_TO_STATE:
        return AlignmentPolicy.EVENT_TRANSFORM
    return AlignmentPolicy.NO_DEFAULT_FILL


def _point_time_param(endpoint: EndpointSpec) -> str | None:
    for name, spec in endpoint.date_params.items():
        if spec.get("kind") == "point":
            return name
    return None


def _range_start_param(endpoint: EndpointSpec) -> str | None:
    for name, spec in endpoint.date_params.items():
        if spec.get("kind") == "range_start":
            return name
    return None


def _range_end_param(endpoint: EndpointSpec) -> str | None:
    for name, spec in endpoint.date_params.items():
        if spec.get("kind") == "range_end":
            return name
    return None


def _visible_time_semantics(source: FieldSourceSpec) -> str:
    if source.frequency is Frequency.DAILY and source.visible_time_column == "trade_date":
        return "same-day market data visibility"
    if source.fill_policy is FillPolicy.ASOF_SNAPSHOT:
        return "visible after source announcement or publication date"
    if source.fill_policy is FillPolicy.NO_FILL:
        return "not safe for default PIT filling"
    return "source-declared PIT visibility"


def _capability_limitations(source: FieldSourceSpec, endpoint: EndpointSpec) -> list[str]:
    limitations: list[str] = []
    if not source.pit_safe:
        limitations.append("pit_visibility_not_declared")
    if not endpoint.supports_marketwide_by_date and not endpoint.supports_symbol_range:
        limitations.append("no_structured_fetch_shape_declared")
    return limitations


def _coverage_blockers(
    requirement: FactorInputRequirement,
    *,
    field_coverage: float,
    time_coverage: float,
    entity_coverage: float,
    matched_source: SourceCapability | None,
) -> list[str]:
    blockers: list[str] = []
    if matched_source is None:
        blockers.append("source_capability_unresolved")
    if field_coverage < requirement.coverage_policy.min_required_field_coverage:
        blockers.append("field_coverage_below_requirement")
    if entity_coverage < requirement.coverage_policy.min_cross_sectional_coverage:
        blockers.append("entity_coverage_below_requirement")
    min_time = requirement.coverage_policy.min_time_coverage
    if min_time is not None and time_coverage < min_time:
        blockers.append("time_coverage_below_requirement")
    return blockers


def _coverage_summary(by_date: dict[str, float]) -> dict[str, float]:
    if not by_date:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0}
    values = pd.Series(list(by_date.values()), dtype="float64")
    return {
        "min": float(values.min()),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
    }


def _estimated_requests(
    shape: FetchShape,
    missing_time_points: list[str],
    entity_ids: list[str],
) -> int:
    if shape.name is FetchShapeName.MARKETWIDE_TIME_SLICE:
        return len(missing_time_points)
    if shape.name is FetchShapeName.SYMBOL_TIME_RANGE:
        return len(entity_ids)
    return 1 if missing_time_points or entity_ids else 0


def _format_date(value: Any) -> str:
    if isinstance(value, date):
        return f"{value:%Y-%m-%d}"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return f"{parsed.date():%Y-%m-%d}"
