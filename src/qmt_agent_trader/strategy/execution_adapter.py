"""Strategy-level adapter over the existing factor-rank research runner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pydantic import BaseModel, Field

from qmt_agent_trader.backtest.research_models import FactorRankResearchResult
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.data.field_sources import FieldSourceIndex, fetch_columns_for_source
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.providers.base import FetchItem
from qmt_agent_trader.data.providers.tushare.planner import TushareFetchPlanner
from qmt_agent_trader.data.providers.tushare.registry import default_tushare_registry
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.trading_calendar import load_open_sessions
from qmt_agent_trader.factors.input_panel import build_target_frequency_panel
from qmt_agent_trader.factors.registry import FactorRegistry, SavedFactor
from qmt_agent_trader.persistence.artifacts import ArtifactMetadata, artifact_store_for_root
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.strategy.adapter_capabilities import validate_factor_rank_adapter_spec
from qmt_agent_trader.strategy.diagnostics import StrategyDiagnosticsEvaluator
from qmt_agent_trader.strategy.models import FactorLeg, SavedStrategy, StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry

BACKTEST_BASE_FIELDS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "suspended",
    "limit_up",
    "limit_down",
    "st",
]
MIN_REQUIRED_FIELD_COVERAGE = 0.80
MIN_CROSS_SECTIONAL_COVERAGE = 0.50


class StrategyBacktestConfig(BaseModel):
    strategy_id: str
    start_date: str
    end_date: str
    universe: str = "stock_etf"
    initial_cash: float = 1_000_000
    execution_delay_days: int = 1
    slippage_bps: float = 5.0
    top_n: int = 20
    max_single_position_pct: float = 0.10
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "daily"
    min_turnover_threshold: float = Field(default=0.0, ge=0, le=1)
    rank_buffer: int = Field(default=0, ge=0)
    cash_buffer_pct: float = Field(default=0.02, ge=0, lt=1)
    lower_is_better: bool = False
    min_daily_cross_sectional_coverage: float = Field(default=0.80, gt=0, le=1)
    min_reference_symbols_for_coverage_gate: int = Field(default=50, ge=1)
    symbols: list[str] = Field(default_factory=list)
    symbols_by_date: dict[str, list[str]] | None = None
    universe_mode: Literal["snapshot", "rolling"] = "snapshot"
    strategy_spec: StrategySpec | None = None
    implementation_code_path: str | None = None
    factor_name: str | None = None


class StrategyBacktestResult(BaseModel):
    run_id: str
    strategy_id: str
    strategy_version: str
    status: str
    metrics: dict[str, object] = Field(default_factory=dict)
    report_path: str | None = None
    leakage_report: dict[str, object] = Field(default_factory=dict)
    diagnostics: dict[str, object] | None = None
    message: str | None = None
    factor_ids: list[str] = Field(default_factory=list)
    execution_backend: str = "factor_rank_baseline_adapter"
    research_only: bool = True
    live_trading_allowed: bool = False
    data_window: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    adapter_limitations: list[str] = Field(default_factory=list)
    requested_factor_ids: list[str] = Field(default_factory=list)
    factor_id: str | None = None
    composite_method: str | None = None
    factor_weights: dict[str, float] = Field(default_factory=dict)
    factor_directions: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    data_requirements: dict[str, object] = Field(default_factory=dict)
    input_panel_metadata: dict[str, object] = Field(default_factory=dict)
    missing_columns: list[str] = Field(default_factory=list)
    available_columns: list[str] = Field(default_factory=list)
    coverage_by_field: dict[str, object] = Field(default_factory=dict)
    field_sources: dict[str, object] = Field(default_factory=dict)
    unresolved_fields: list[object] = Field(default_factory=list)
    missing_fields: dict[str, object] = Field(default_factory=dict)
    next_repair_tool: str | None = None
    suggested_repair: dict[str, object] = Field(default_factory=dict)
    unsupported_fields: list[str] = Field(default_factory=list)
    capability_issues: list[dict[str, object]] = Field(default_factory=list)
    schema_version: str | None = None
    equity_points: list[dict[str, object]] = Field(default_factory=list)
    rebalance_points: list[dict[str, object]] = Field(default_factory=list)
    trade_blotter: list[dict[str, object]] = Field(default_factory=list)
    data_quality: dict[str, object] = Field(default_factory=dict)
    cost_attribution: dict[str, object] = Field(default_factory=dict)


def run_strategy_backtest(
    lake: DataLake,
    registry: StrategyRegistry,
    config: StrategyBacktestConfig,
    *,
    reports_dir: Path,
) -> StrategyBacktestResult:
    run_id = new_id("research")
    saved_strategy = _strategy_from_registry(registry, config.strategy_id)
    spec = config.strategy_spec or (saved_strategy.spec if saved_strategy is not None else None)
    effective_code_path = config.implementation_code_path or (
        saved_strategy.code_path if saved_strategy is not None else None
    )
    if spec is not None:
        capability_issues = validate_factor_rank_adapter_spec(
            spec,
            code_path=effective_code_path,
        )
        if capability_issues:
            generated_code = any(issue.field == "code_path" for issue in capability_issues)
            return StrategyBacktestResult(
                run_id=run_id,
                strategy_id=config.strategy_id,
                strategy_version=spec.version,
                status="BLOCKED",
                reason=(
                    "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"
                    if generated_code
                    else "UNSUPPORTED_STRATEGY_SEMANTICS"
                ),
                unsupported_fields=[issue.field for issue in capability_issues],
                capability_issues=[asdict(issue) for issue in capability_issues],
            )
    factor_name = config.factor_name or _first_factor_id(spec)
    requested_factor_ids = _factor_ids(spec) or ([factor_name] if factor_name else [])
    composite_method = _composite_method(spec)
    factor_weights = _factor_weights(spec)
    factor_directions = _factor_directions(spec)
    used_factor_name, factor_ids, factor_registry, execution_backend = _execution_factor_registry(
        lake,
        config,
        spec,
        factor_name,
        requested_factor_ids,
    )
    adapter_limitations = _adapter_limitations(requested_factor_ids, used_factor_name)
    warnings = list(adapter_limitations)
    if not used_factor_name:
        return _error(
            run_id,
            config,
            spec,
            "FACTOR_NOT_FOUND",
            "strategy has no factor to backtest",
        )

    load_symbols = config.symbols or _symbols_from_date_map(config.symbols_by_date)
    factor_input_fields = _required_fields_for_backtest_factors(
        factor_registry=factor_registry,
        requested_factor_ids=requested_factor_ids,
    )
    panel, panel_metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start=config.start_date,
        target_end=config.end_date,
        required_fields=factor_input_fields,
        symbols=load_symbols or None,
    )
    blocked = _blocked_backtest_from_panel_metadata(
        run_id=run_id,
        config=config,
        spec=spec,
        requested_factor_ids=requested_factor_ids,
        factor_registry=factor_registry,
        panel=panel,
        panel_metadata=panel_metadata,
    )
    if blocked is not None:
        return blocked
    warnings.extend(
        _partial_coverage_warnings(
            panel_metadata=panel_metadata,
            factor_registry=factor_registry,
            requested_factor_ids=requested_factor_ids,
        )
    )
    if panel.empty:
        return _error(
            run_id,
            config,
            spec,
            "DATA_NOT_READY",
            "no factor input panel rows available for requested range",
        )
    actual_start = panel["trade_date"].min()
    actual_end = panel["trade_date"].max()
    data_window = {
        "requested_start": config.start_date,
        "requested_end": config.end_date,
        "actual_start": f"{actual_start:%Y%m%d}",
        "actual_end": f"{actual_end:%Y%m%d}",
        "data_freshness": (
            "stale_vs_requested_end"
            if actual_end < pd.to_datetime(config.end_date).date()
            else "covers_requested_end"
        ),
    }
    scenario = SensitivityScenario(
        cost_multiplier=1.0,
        slippage_bps=config.slippage_bps,
        execution_delay_days=config.execution_delay_days,
        top_n=config.top_n,
        max_single_position_pct=config.max_single_position_pct,
    )
    expected_trade_dates = load_open_sessions(
        lake,
        start=config.start_date,
        end=config.end_date,
    )
    # TODO: preload factor lookback window before config.start_date, then trim signals.
    runner = FactorRankResearchRunner(
        panel,
        FactorRankResearchConfig(
            factor_name=used_factor_name,
            expected_trade_dates=expected_trade_dates,
            factor_registry_root=lake.root.parent / "factors",
            factor_registry=factor_registry,
            top_n=config.top_n,
            max_single_position_pct=config.max_single_position_pct,
            initial_cash=config.initial_cash,
            rebalance_frequency=config.rebalance_frequency,
            min_turnover_threshold=config.min_turnover_threshold,
            rank_buffer=config.rank_buffer,
            cash_buffer_pct=config.cash_buffer_pct,
            lower_is_better=config.lower_is_better,
            symbols_by_date=config.symbols_by_date,
        ),
    )
    result = runner.run(scenario)

    result_dict = result.as_dict()
    metrics = _build_canonical_metrics(result, config)
    leakage_report: dict[str, object] = {
        "valid": True,
        "execution_delay_days": config.execution_delay_days,
    }
    evidence = _diagnostic_evidence(
        result_dict,
        leakage_report,
        canonical_metrics=metrics,
        factor_frame=runner.factor_frame,
        bars=runner.bars,
        initial_cash=config.initial_cash,
    )
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence).as_dict()
    canonical_evidence = _canonical_result_evidence(result_dict, metrics)
    report = {
        "schema_version": "2.0",
        "run_id": run_id,
        "created_at": shanghai_now_iso(),
        "artifact_type": "strategy_backtest",
        "strategy_id": config.strategy_id,
        "strategy_version": spec.version if spec else "0.1.0",
        "requested_factor_ids": requested_factor_ids,
        "factor_ids": factor_ids,
        "execution_backend": execution_backend,
        "composite_method": composite_method,
        "factor_weights": factor_weights,
        "factor_directions": factor_directions,
        "research_only": True,
        "live_trading_allowed": False,
        "warnings": warnings,
        "adapter_limitations": adapter_limitations,
        "config": config.model_dump(mode="json"),
        "data_window": data_window,
        "input_panel_metadata": panel_metadata,
        "metrics": metrics,
        "leakage_report": leakage_report,
        "diagnostics": diagnostics,
        **canonical_evidence,
        "payload": result_dict,
    }
    receipt = artifact_store_for_root(reports_dir, lock_manager=lake.lock_manager).create(
        f"{run_id}.json",
        json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        metadata=ArtifactMetadata(
            artifact_id=run_id,
            artifact_type="strategy_backtest_report",
            producer="strategy.execution_adapter.run_strategy_backtest",
            related_run_id=run_id,
            related_strategy_id=config.strategy_id,
            related_factor_id=factor_ids[0] if len(factor_ids) == 1 else None,
        ),
    )
    report_path = receipt.path
    return StrategyBacktestResult(
        run_id=run_id,
        strategy_id=config.strategy_id,
        strategy_version=spec.version if spec else "0.1.0",
        status="completed",
        metrics=metrics,
        report_path=str(report_path),
        leakage_report=leakage_report,
        diagnostics=diagnostics,
        requested_factor_ids=requested_factor_ids,
        factor_ids=factor_ids,
        execution_backend=execution_backend,
        composite_method=composite_method,
        factor_weights=factor_weights,
        factor_directions=factor_directions,
        data_window=data_window,
        input_panel_metadata=panel_metadata,
        coverage_by_field=dict(panel_metadata.get("coverage_by_field") or {}),
        field_sources=dict(panel_metadata.get("field_sources") or {}),
        unresolved_fields=list(panel_metadata.get("unresolved_fields") or []),
        missing_fields=dict(panel_metadata.get("missing_fields") or {}),
        warnings=warnings,
        adapter_limitations=adapter_limitations,
        schema_version="2.0",
        equity_points=canonical_evidence["equity_points"],
        rebalance_points=canonical_evidence["rebalance_points"],
        trade_blotter=canonical_evidence["trade_blotter"],
        data_quality=canonical_evidence["data_quality"],
        cost_attribution=canonical_evidence["cost_attribution"],
    )


def _build_canonical_metrics(
    result: FactorRankResearchResult,
    config: StrategyBacktestConfig,
) -> dict[str, object]:
    net_return = result.metrics.total_return
    gross_return = result.same_trade_gross_return
    return {
        "total_return": round(net_return, 6),
        "net_total_return": round(net_return, 6),
        "same_trade_gross_return": round(gross_return, 6),
        "cost_drag": round(gross_return - net_return, 6),
        "sharpe": round(result.metrics.sharpe, 6),
        "max_drawdown": round(result.metrics.max_drawdown, 6),
        "turnover": round(result.metrics.turnover, 6),
        "average_one_way_turnover": round(result.metrics.turnover, 6),
        "average_top_n_overlap": round(result.average_top_n_overlap, 6),
        "explicit_cost_to_initial_cash": round(
            result.total_explicit_cost / config.initial_cash,
            6,
        ),
        "slippage_cost_to_initial_cash": round(
            result.total_slippage_cost / config.initial_cash,
            6,
        ),
        "trade_count": len(result.trades),
    }


def _canonical_result_evidence(
    result_dict: dict[str, Any],
    metrics: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "equity_points": list(result_dict.get("equity_points") or []),
        "rebalance_points": list(result_dict.get("rebalance_points") or []),
        "trade_blotter": list(result_dict.get("trades") or []),
        "data_quality": dict(result_dict.get("data_quality") or {}),
        "cost_attribution": {
            "explicit_cost": float(result_dict.get("total_explicit_cost") or 0.0),
            "slippage_cost": float(result_dict.get("total_slippage_cost") or 0.0),
            "same_trade_gross_return": metrics.get("same_trade_gross_return", 0.0),
            "net_total_return": metrics.get("net_total_return", 0.0),
            "cost_drag": metrics.get("cost_drag", 0.0),
        },
    }


def _strategy_from_registry(
    registry: StrategyRegistry,
    strategy_id: str,
) -> SavedStrategy | None:
    return registry.get_strategy(strategy_id)


def _first_factor_id(spec: StrategySpec | None) -> str | None:
    if spec is None or not spec.factors:
        return None
    return spec.factors[0].factor_id


def _symbols_from_date_map(symbols_by_date: dict[str, list[str]] | None) -> list[str]:
    if not symbols_by_date:
        return []
    symbols: list[str] = []
    for date_symbols in symbols_by_date.values():
        for symbol in date_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _factor_ids(spec: StrategySpec | None) -> list[str]:
    if spec is None:
        return []
    return [factor.factor_id for factor in spec.factors]


def _composite_method(spec: StrategySpec | None) -> str | None:
    if spec is None or len(spec.factors) <= 1:
        return None
    return "trade_date_cross_sectional_zscore_weighted_sum"


def _factor_weights(spec: StrategySpec | None) -> dict[str, float]:
    if spec is None:
        return {}
    return {factor.factor_id: float(factor.weight) for factor in spec.factors}


def _factor_directions(spec: StrategySpec | None) -> dict[str, str]:
    if spec is None:
        return {}
    return {
        factor.factor_id: "ascending" if factor.ascending else "descending"
        for factor in spec.factors
    }


def _adapter_limitations(requested_factor_ids: list[str], used_factor_id: str | None) -> list[str]:
    if len(requested_factor_ids) <= 1 or used_factor_id is None:
        return []
    if used_factor_id.startswith("__composite__"):
        return []
    return [
        (
            "factor_rank_baseline_adapter did not execute all requested factors; "
            f"used '{used_factor_id}' from requested factors {requested_factor_ids}."
        )
    ]


def _execution_factor_registry(
    lake: DataLake,
    config: StrategyBacktestConfig,
    spec: StrategySpec | None,
    factor_name: str | None,
    requested_factor_ids: list[str],
) -> tuple[str | None, list[str], FactorRegistry | None, str]:
    base_registry = FactorRegistry(
        lake.root.parent / "factors",
        lock_manager=lake.lock_manager,
        atomic_store=AtomicFileStore(lake.lock_manager),
    )
    if spec is not None and len(spec.factors) > 1:
        composite_name = f"__composite__{config.strategy_id}"
        return (
            composite_name,
            requested_factor_ids,
            _CompositeFactorRegistry(base_registry, composite_name, spec.factors),
            "factor_rank_composite_adapter",
        )
    return (
        factor_name,
        ([factor_name] if factor_name else []),
        base_registry,
        "factor_rank_baseline_adapter",
    )


def _required_fields_for_backtest_factors(
    *,
    factor_registry: FactorRegistry | None,
    requested_factor_ids: list[str],
) -> list[str]:
    fields = list(BACKTEST_BASE_FIELDS)
    if factor_registry is None:
        return fields
    for factor_id in requested_factor_ids:
        saved = factor_registry.get_factor(factor_id)
        if saved is None:
            continue
        for column in saved.required_columns:
            if column not in fields:
                fields.append(column)
    return fields


def _blocked_backtest_from_panel_metadata(
    *,
    run_id: str,
    config: StrategyBacktestConfig,
    spec: StrategySpec | None,
    requested_factor_ids: list[str],
    factor_registry: FactorRegistry | None,
    panel: pd.DataFrame,
    panel_metadata: dict[str, Any],
) -> StrategyBacktestResult | None:
    abrupt_dates = _abrupt_low_coverage_dates(panel_metadata, config)
    panel_metadata["abrupt_low_coverage_dates"] = abrupt_dates
    if abrupt_dates:
        panel_metadata["input_panel_status"] = "ABRUPT_DAILY_COVERAGE_DROP"
        panel_metadata["evidence_status"] = "BLOCKED"
        return StrategyBacktestResult(
            run_id=run_id,
            strategy_id=config.strategy_id,
            strategy_version=spec.version if spec else "0.1.0",
            status="BLOCKED",
            reason="ABRUPT_DAILY_COVERAGE_DROP",
            message=f"daily market-data cross-section collapsed on {abrupt_dates}",
            requested_factor_ids=requested_factor_ids,
            factor_ids=requested_factor_ids,
            input_panel_metadata=panel_metadata,
            diagnostics={
                "status": "BLOCKED",
                "checks": [{
                    "name": "daily_cross_sectional_coverage",
                    "status": "BLOCKED",
                    "observed": abrupt_dates,
                    "threshold": config.min_daily_cross_sectional_coverage,
                    "message": "abrupt daily symbol coverage drop blocks execution",
                    "evidence_source": "input_panel_metadata",
                }],
            },
        )
    missing_columns = _missing_factor_input_columns(
        panel=panel,
        factor_registry=factor_registry,
        requested_factor_ids=requested_factor_ids,
    )
    panel_status = str(panel_metadata.get("status") or "")
    unresolved = _unresolved_fields_for_columns(panel_metadata, missing_columns)
    missing_fields = _missing_fields_for_columns(panel_metadata, missing_columns)
    low_coverage_fields = _low_coverage_required_fields(
        panel_metadata=panel_metadata,
        factor_registry=factor_registry,
        requested_factor_ids=requested_factor_ids,
    )
    no_data = panel.empty or panel_status == "NO_DATA"
    should_block = bool(
        no_data or missing_columns or unresolved or missing_fields or low_coverage_fields
    )
    if not should_block:
        panel_metadata.setdefault("input_panel_status", panel_status or "OK")
        panel_metadata.setdefault("evidence_status", "STRONG")
        return None

    reason = (
        "INPUT_PANEL_NO_DATA"
        if no_data
        else "INPUT_PANEL_PARTIAL_COVERAGE"
        if panel_status == "PARTIAL_COVERAGE" or low_coverage_fields
        else "MISSING_FACTOR_INPUTS"
    )
    status = "DATA_NOT_READY" if no_data else "BLOCKED"
    repair_columns = missing_columns or list(low_coverage_fields)
    suggested_repair = _suggest_repair_for_factor_input_metadata(
        panel_metadata,
        missing_columns=repair_columns,
        start_date=config.start_date,
        end_date=config.end_date,
        symbols=config.symbols or _symbols_from_date_map(config.symbols_by_date),
        requested_factor_ids=requested_factor_ids,
        strategy_id=config.strategy_id,
        factor_name=config.factor_name,
    )
    message = (
        "factor input panel has no rows for requested range"
        if no_data
        else (
            "factor input panel has required fields below coverage threshold: "
            f"{low_coverage_fields}"
        )
        if low_coverage_fields
        else f"factor input panel is missing required columns: {missing_columns}"
    )
    panel_metadata["input_panel_status"] = panel_status or "PARTIAL_COVERAGE"
    panel_metadata["evidence_status"] = "BLOCKED"
    return StrategyBacktestResult(
        run_id=run_id,
        strategy_id=config.strategy_id,
        strategy_version=spec.version if spec else "0.1.0",
        status=status,
        reason=reason,
        message=message,
        requested_factor_ids=requested_factor_ids,
        factor_id=requested_factor_ids[0] if requested_factor_ids else config.factor_name,
        factor_ids=requested_factor_ids,
        execution_backend=(
            "factor_rank_composite_adapter"
            if len(requested_factor_ids) > 1
            else "factor_rank_baseline_adapter"
        ),
        research_only=True,
        live_trading_allowed=False,
        data_window={
            "requested_start": config.start_date,
            "requested_end": config.end_date,
            "symbols": config.symbols,
        },
        data_requirements={
            "required_fields": list(panel_metadata.get("required_fields") or []),
            "requested_factor_ids": requested_factor_ids,
            "min_required_field_coverage": MIN_REQUIRED_FIELD_COVERAGE,
            "min_cross_sectional_coverage": MIN_CROSS_SECTIONAL_COVERAGE,
        },
        input_panel_metadata=panel_metadata,
        missing_columns=repair_columns,
        available_columns=sorted(str(column) for column in panel.columns),
        coverage_by_field=dict(panel_metadata.get("coverage_by_field") or {}),
        field_sources=dict(panel_metadata.get("field_sources") or {}),
        unresolved_fields=list(panel_metadata.get("unresolved_fields") or []),
        missing_fields=dict(panel_metadata.get("missing_fields") or {}),
        next_repair_tool="run_tushare_fetch",
        suggested_repair=suggested_repair,
        warnings=list(panel_metadata.get("warnings") or []),
        diagnostics={
            "status": "BLOCKED",
            "checks": [
                {
                    "name": "factor_input_panel",
                    "status": "BLOCKED",
                    "observed": {
                        "missing_columns": missing_columns,
                        "low_coverage_fields": low_coverage_fields,
                    },
                    "threshold": {
                        "required_field_coverage": MIN_REQUIRED_FIELD_COVERAGE,
                        "cross_sectional_coverage": MIN_CROSS_SECTIONAL_COVERAGE,
                    },
                    "message": message,
                    "evidence_source": "input_panel_metadata",
                }
            ],
        },
    )


def _abrupt_low_coverage_dates(
    panel_metadata: dict[str, Any],
    config: StrategyBacktestConfig,
) -> list[str]:
    ratios = panel_metadata.get("daily_cross_sectional_coverage")
    references = panel_metadata.get("daily_reference_symbol_counts")
    if not isinstance(ratios, dict) or not isinstance(references, dict):
        return []
    minimum_reference = max(config.min_reference_symbols_for_coverage_gate, config.top_n * 2)
    return sorted(
        str(day)
        for day, ratio in ratios.items()
        if float(references.get(day, 0.0)) >= minimum_reference
        and float(ratio) < config.min_daily_cross_sectional_coverage
    )


def _missing_factor_input_columns(
    *,
    panel: pd.DataFrame,
    factor_registry: FactorRegistry | None,
    requested_factor_ids: list[str],
) -> list[str]:
    if factor_registry is None:
        return []
    missing: list[str] = []
    for factor_id in requested_factor_ids:
        saved = factor_registry.get_factor(factor_id)
        if saved is None:
            continue
        for column in saved.required_columns:
            if column in {"symbol", "trade_date"}:
                continue
            if column not in panel.columns:
                if column not in missing:
                    missing.append(column)
                continue
            if panel[column].isna().all() and column not in missing:
                missing.append(column)
    return missing


def _low_coverage_required_fields(
    *,
    panel_metadata: dict[str, Any],
    factor_registry: FactorRegistry | None,
    requested_factor_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if factor_registry is None:
        return {}
    coverage = panel_metadata.get("coverage_by_field")
    coverage = coverage if isinstance(coverage, dict) else {}
    low: dict[str, dict[str, Any]] = {}
    required_fields = _required_factor_fields_only(
        factor_registry=factor_registry,
        requested_factor_ids=requested_factor_ids,
    )
    for field in required_fields:
        item = coverage.get(field)
        if not isinstance(item, dict):
            continue
        value = float(item.get("coverage") or 0.0)
        if 0.0 < value < MIN_REQUIRED_FIELD_COVERAGE:
            low[field] = {
                "coverage": value,
                "threshold": MIN_REQUIRED_FIELD_COVERAGE,
                "non_null_rows": int(item.get("non_null_rows") or item.get("non_null") or 0),
                "total_rows": int(item.get("total_rows") or 0),
                "source": item.get("source"),
                "join_policy": item.get("join_policy"),
                "pit_safe": item.get("pit_safe"),
            }
    return low


def _required_factor_fields_only(
    *,
    factor_registry: FactorRegistry,
    requested_factor_ids: list[str],
) -> list[str]:
    fields: list[str] = []
    for factor_id in requested_factor_ids:
        saved = factor_registry.get_factor(factor_id)
        if saved is None:
            continue
        for column in saved.required_columns:
            if column in {"symbol", "trade_date"} or column in BACKTEST_BASE_FIELDS:
                continue
            if column not in fields:
                fields.append(column)
    return fields


def _partial_coverage_warnings(
    *,
    panel_metadata: dict[str, Any],
    factor_registry: FactorRegistry | None,
    requested_factor_ids: list[str],
) -> list[str]:
    if factor_registry is None:
        return []
    warnings: list[str] = []
    coverage = panel_metadata.get("coverage_by_field")
    coverage = coverage if isinstance(coverage, dict) else {}
    for field in _required_fields_for_backtest_factors(
        factor_registry=factor_registry,
        requested_factor_ids=requested_factor_ids,
    ):
        if field in {"symbol", "trade_date"}:
            continue
        item = coverage.get(field)
        if not isinstance(item, dict):
            continue
        value = float(item.get("coverage") or 0.0)
        if 0.0 < value < 1.0:
            warnings.append(f"input_panel_partial_coverage:{field}:{value:.4f}")
    return warnings


def _unresolved_fields_for_columns(
    panel_metadata: dict[str, Any],
    columns: list[str],
) -> list[dict[str, Any]]:
    wanted = set(columns)
    return [
        item
        for item in panel_metadata.get("unresolved_fields") or []
        if isinstance(item, dict) and str(item.get("field")) in wanted
    ]


def _missing_fields_for_columns(
    panel_metadata: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    wanted = set(columns)
    missing = panel_metadata.get("missing_fields")
    if not isinstance(missing, dict):
        return {}
    return {field: value for field, value in missing.items() if field in wanted}


def _suggest_repair_for_factor_input_metadata(
    metadata: dict[str, Any],
    *,
    missing_columns: list[str],
    start_date: str,
    end_date: str,
    symbols: list[str],
    requested_factor_ids: list[str],
    strategy_id: str,
    factor_name: str | None,
) -> dict[str, Any]:
    registry = default_tushare_registry()
    source_index = FieldSourceIndex.from_tushare_registry(registry)
    planner = TushareFetchPlanner(registry)
    fields_by_api: dict[str, list[str]] = {}
    sources_by_api: dict[str, Any] = {}
    for field in missing_columns:
        source = source_index.best_source_for_field(field, target_frequency=Frequency.DAILY)
        if source is None:
            field_sources = metadata.get("field_sources")
            field_source = field_sources.get(field) if isinstance(field_sources, dict) else None
            if isinstance(field_source, dict):
                api_name = str(field_source.get("api_name") or "")
                if api_name:
                    fields_by_api.setdefault(api_name, []).append(field)
            continue
        fields_by_api.setdefault(source.api_name, []).append(field)
        sources_by_api[source.api_name] = source

    items: list[dict[str, Any]] = []
    for api_name, fields in fields_by_api.items():
        source = sources_by_api.get(api_name)
        raw_fields = (
            fetch_columns_for_source(source, fields)
            if source is not None
            else list(dict.fromkeys(fields))
        )
        spec = registry.get(api_name)
        plan = planner.plan(
            [
                FetchItem(
                    api_name=api_name,
                    symbols=symbols,
                    fields=raw_fields,
                    start_date=start_date,
                    end_date=end_date,
                )
            ]
        )
        item: dict[str, Any] = {
            "api_name": api_name,
            "symbols": symbols,
            "fields": raw_fields,
            "start_date": start_date,
            "end_date": end_date,
            "required_fields": fields,
            "symbols_count": len(symbols),
            "estimated_request_count": plan.estimated_request_count,
            "planner_status": plan.status,
            "planner_reason": plan.reason,
            "requires_manual_approval": True,
        }
        if source is not None:
            item["raw_dataset_name"] = source.raw_dataset_name
            item["target_dataset"] = f"raw/{source.raw_dataset_name}"
        if spec is not None:
            item["endpoint_capability"] = {
                "supports_symbol_range": spec.supports_symbol_range,
                "supports_marketwide_by_date": spec.supports_marketwide_by_date,
                "rows_per_request": spec.call_limit.get("rows_per_request"),
            }
        if plan.errors:
            item["planner_errors"] = plan.errors
        items.append(item)

    return {
        "tool": "run_tushare_fetch",
        "execute_plan": True,
        "items": items,
        "missing_columns": missing_columns,
        "then_run": [
            "build_data_table(daily_market)",
            "run_backtest(...)",
        ],
        "verification_action": {
            "tool": "run_backtest",
            "args": {
                "strategy_id": strategy_id,
                "factor_name": factor_name,
                "requested_factor_ids": requested_factor_ids,
                "start_date": start_date,
                "end_date": end_date,
                "symbols": symbols,
            },
        },
    }


def _diagnostic_evidence(
    result_dict: dict[str, Any],
    leakage_report: dict[str, object],
    *,
    canonical_metrics: Mapping[str, object],
    factor_frame: pd.DataFrame,
    bars: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    observation_count = len(factor_frame)
    non_null = (
        int(factor_frame["factor_value"].notna().sum())
        if "factor_value" in factor_frame
        else 0
    )
    coverage = non_null / observation_count if observation_count else 0.0
    trades = result_dict.get("trades", [])
    trade_count = len(trades) if isinstance(trades, list) else 0
    rejected_orders = int(result_dict.get("rejected_orders", 0) or 0)
    factor_report: dict[str, Any] = {
        "observation_count": observation_count,
        "coverage": coverage,
    }
    factor_report.update(_factor_predictive_report(factor_frame, bars))
    return {
        "leakage_report": leakage_report,
        "data_quality": {
            "abrupt_low_coverage_dates": result_dict.get("data_quality", {}).get(
                "low_cross_section_dates", []
            )
            if isinstance(result_dict.get("data_quality"), dict)
            else [],
        },
        "factor_report": factor_report,
        "performance_report": {
            "max_drawdown": canonical_metrics["max_drawdown"],
        },
        "trade_blotter": trades,
        "turnover_report": {
            "average_turnover": canonical_metrics["average_one_way_turnover"],
        },
        "cost_report": {
            "cost_to_initial_cash": (
                float(cast(float | int, canonical_metrics["explicit_cost_to_initial_cash"]))
                + float(cast(float | int, canonical_metrics["slippage_cost_to_initial_cash"]))
            ),
            "cost_drag": canonical_metrics["cost_drag"],
        },
        "churn_report": {
            "average_top_n_overlap": canonical_metrics["average_top_n_overlap"],
        },
        "rejection_report": {
            "rate": rejected_orders / (trade_count + rejected_orders)
            if trade_count + rejected_orders
            else 0.0
        },
    }


def _factor_predictive_report(factor_frame: pd.DataFrame, bars: pd.DataFrame) -> dict[str, Any]:
    if factor_frame.empty or bars.empty:
        return {}
    required_factor_columns = {"symbol", "trade_date", "factor_value"}
    if not required_factor_columns.issubset(factor_frame.columns):
        return {}
    required_bar_columns = {"symbol", "trade_date", "close"}
    if not required_bar_columns.issubset(bars.columns):
        return {}

    bar_returns = bars[["symbol", "trade_date", "close"]].copy()
    bar_returns = bar_returns.sort_values(["symbol", "trade_date"])
    close = pd.to_numeric(bar_returns["close"], errors="coerce")
    next_close = close.groupby(bar_returns["symbol"]).shift(-1)
    bar_returns["forward_return_1d"] = next_close / close - 1.0

    joined = factor_frame.merge(
        bar_returns[["symbol", "trade_date", "forward_return_1d"]],
        on=["symbol", "trade_date"],
        how="left",
    )
    joined["factor_value"] = pd.to_numeric(joined["factor_value"], errors="coerce")
    joined["forward_return_1d"] = pd.to_numeric(
        joined["forward_return_1d"],
        errors="coerce",
    )
    joined = joined.dropna(subset=["factor_value", "forward_return_1d"])
    if joined.empty:
        return {}

    ic_by_date: dict[str, float] = {}
    spread_by_date: dict[str, float] = {}
    for trade_date, group in joined.groupby("trade_date"):
        clean = group.dropna(subset=["factor_value", "forward_return_1d"])
        if len(clean) < 2:
            continue
        factor_unique = clean["factor_value"].nunique(dropna=True)
        return_unique = clean["forward_return_1d"].nunique(dropna=True)
        if factor_unique < 2 or return_unique < 2:
            continue
        ic = clean["factor_value"].corr(clean["forward_return_1d"], method="spearman")
        if pd.notna(ic):
            key = _format_trade_date(trade_date)
            ic_by_date[key] = float(ic)
            ranked = clean.sort_values("factor_value")
            bottom = float(ranked.iloc[0]["forward_return_1d"])
            top = float(ranked.iloc[-1]["forward_return_1d"])
            spread_by_date[key] = top - bottom
    if not ic_by_date:
        return {}

    walk_forward = _walk_forward_from_daily_ic(ic_by_date, spread_by_date)
    return {
        "positive_ic_ratio": sum(1 for value in ic_by_date.values() if value > 0)
        / len(ic_by_date),
        "ic_by_date": ic_by_date,
        "walk_forward": walk_forward,
        "evidence_source": "computed_from_factor_frame_forward_returns",
    }


def _walk_forward_from_daily_ic(
    ic_by_date: dict[str, float],
    spread_by_date: dict[str, float],
    *,
    window_days: int = 20,
) -> list[dict[str, Any]]:
    dates = sorted(ic_by_date)
    if not dates:
        return []
    slices: list[dict[str, Any]] = []
    for start_index in range(0, len(dates), window_days):
        window = dates[start_index : start_index + window_days]
        if not window:
            continue
        ic_values = [ic_by_date[item] for item in window]
        spread_values = [
            spread_by_date[item] for item in window if item in spread_by_date
        ]
        slices.append(
            {
                "start": window[0],
                "end": window[-1],
                "observations": len(window),
                "non_null": len(window),
                "coverage": 1.0,
                "mean_ic": sum(ic_values) / len(ic_values),
                "long_short_spread": (
                    sum(spread_values) / len(spread_values) if spread_values else None
                ),
            }
        )
    return slices


def _format_trade_date(value: Any) -> str:
    try:
        timestamp = pd.to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    if pd.isna(timestamp):
        return str(value)
    return f"{timestamp:%Y%m%d}"


class _CompositeFactorRegistry(FactorRegistry):
    def __init__(
        self,
        base: FactorRegistry,
        composite_name: str,
        factors: list[FactorLeg],
    ) -> None:
        self.base = base
        self.composite_name = composite_name
        self.factors = factors

    def get_factor(self, factor_id: str) -> SavedFactor | None:
        if factor_id == self.composite_name:
            return SavedFactor(
                factor_id=self.composite_name,
                name=self.composite_name,
                version="0.1.0",
                implementation_ref="memory:composite",
                required_columns=("symbol", "trade_date", "close"),
                lookback=0,
                params={},
                created_by="adapter",
                created_at="runtime",
            )
        return self.base.get_factor(factor_id)

    def compute(self, factor_id: str, bars: pd.DataFrame) -> pd.Series:
        if factor_id != self.composite_name:
            return self.base.compute(factor_id, bars)
        score = pd.Series(0.0, index=bars.index, dtype="float64")
        for leg in self.factors:
            values = pd.to_numeric(self.base.compute(leg.factor_id, bars), errors="coerce")
            normalized = values.groupby(bars["trade_date"]).transform(_zscore)
            if leg.ascending:
                normalized = -normalized
            score = score + normalized.fillna(0.0) * float(leg.weight)
        return score


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or float(std) == 0.0:
        return values * 0.0
    return (values - values.mean()) / std


def _error(
    run_id: str,
    config: StrategyBacktestConfig,
    spec: StrategySpec | None,
    status: str,
    message: str,
) -> StrategyBacktestResult:
    return StrategyBacktestResult(
        run_id=run_id,
        strategy_id=config.strategy_id,
        strategy_version=spec.version if spec else "0.1.0",
        status=status,
        message=message,
    )
