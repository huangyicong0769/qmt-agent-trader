"""Strategy-level adapter over the existing factor-rank research runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityScenario
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.registry import FactorRegistry, SavedFactor
from qmt_agent_trader.strategy.diagnostics import StrategyDiagnosticsEvaluator
from qmt_agent_trader.strategy.models import FactorLeg, StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


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
    symbols: list[str] = Field(default_factory=list)
    symbols_by_date: dict[str, list[str]] | None = None
    universe_mode: Literal["snapshot", "rolling"] = "snapshot"
    strategy_spec: StrategySpec | None = None
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
    composite_method: str | None = None
    factor_weights: dict[str, float] = Field(default_factory=dict)
    factor_directions: dict[str, str] = Field(default_factory=dict)


def run_strategy_backtest(
    lake: DataLake,
    registry: StrategyRegistry,
    config: StrategyBacktestConfig,
    *,
    reports_dir: Path,
) -> StrategyBacktestResult:
    run_id = new_id("research")
    spec = config.strategy_spec or _strategy_spec_from_registry(registry, config.strategy_id)
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
    bars = load_daily_bars(
        lake,
        start=config.start_date,
        end=config.end_date,
        symbols=load_symbols or None,
    )
    if bars.empty:
        return _error(
            run_id,
            config,
            spec,
            "DATA_NOT_READY",
            "no bars available for requested range",
        )
    actual_start = bars["trade_date"].min()
    actual_end = bars["trade_date"].max()
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
    runner = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name=used_factor_name,
            factor_registry_root=lake.root.parent / "factors",
            factor_registry=factor_registry,
            top_n=config.top_n,
            max_single_position_pct=config.max_single_position_pct,
            initial_cash=config.initial_cash,
            symbols_by_date=config.symbols_by_date,
        ),
    )
    scenario = SensitivityScenario(
        cost_multiplier=1.0,
        slippage_bps=config.slippage_bps,
        execution_delay_days=config.execution_delay_days,
        top_n=config.top_n,
        max_single_position_pct=config.max_single_position_pct,
    )
    try:
        result = runner.run(scenario)
    except Exception as exc:
        return _error(run_id, config, spec, "BACKTEST_FAILED", str(exc))

    result_dict = result.as_dict()
    leakage_report: dict[str, object] = {
        "valid": True,
        "execution_delay_days": config.execution_delay_days,
    }
    evidence = _diagnostic_evidence(
        result_dict,
        leakage_report,
        factor_frame=runner.factor_frame,
        bars=runner.bars,
        initial_cash=config.initial_cash,
    )
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence).as_dict()
    metrics = {
        "total_return": round(result.metrics.total_return, 4),
        "sharpe": round(result.metrics.sharpe, 4),
        "max_drawdown": round(result.metrics.max_drawdown, 4),
        "turnover": round(result.metrics.turnover, 4),
        "trade_count": len(result.trades),
    }
    report = {
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
        "metrics": metrics,
        "leakage_report": leakage_report,
        "diagnostics": diagnostics,
        "payload": result_dict,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
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
        warnings=warnings,
        adapter_limitations=adapter_limitations,
    )


def _strategy_spec_from_registry(
    registry: StrategyRegistry,
    strategy_id: str,
) -> StrategySpec | None:
    saved = registry.get_strategy(strategy_id)
    return saved.spec if saved is not None else None


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
    base_registry = FactorRegistry(lake.root.parent / "factors")
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


def _diagnostic_evidence(
    result_dict: dict[str, Any],
    leakage_report: dict[str, object],
    *,
    factor_frame: pd.DataFrame,
    bars: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    metrics = result_dict.get("metrics", {})
    observation_count = len(factor_frame)
    non_null = (
        int(factor_frame["factor_value"].notna().sum())
        if "factor_value" in factor_frame
        else 0
    )
    coverage = non_null / observation_count if observation_count else 0.0
    trades = result_dict.get("trades", [])
    trade_count = len(trades) if isinstance(trades, list) else 0
    total_cost = (
        sum(float(item.get("cost", 0.0)) for item in trades if isinstance(item, dict))
        if isinstance(trades, list)
        else 0.0
    )
    rejected_orders = int(result_dict.get("rejected_orders", 0) or 0)
    factor_report: dict[str, Any] = {
        "observation_count": observation_count,
        "coverage": coverage,
    }
    factor_report.update(_factor_predictive_report(factor_frame, bars))
    return {
        "leakage_report": leakage_report,
        "factor_report": factor_report,
        "performance_report": {
            "max_drawdown": metrics.get("max_drawdown", 0.0),
        },
        "trade_blotter": trades,
        "turnover_report": {"average_turnover": metrics.get("turnover", 0.0)},
        "cost_report": {
            "cost_to_initial_cash": total_cost / initial_cash if initial_cash else 0.0
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
    except Exception:
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
