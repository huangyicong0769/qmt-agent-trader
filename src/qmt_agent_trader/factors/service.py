"""Factor computation service backed by canonical daily bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.context import load_factor_context
from qmt_agent_trader.factors.registry import FactorRegistry


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
    bar_columns = {
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
    }
    if set(saved.required_columns).issubset(bar_columns):
        return load_daily_bars(lake, end=date)
    frame = load_factor_context(lake, as_of_date=date)
    missing = [column for column in saved.required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"factor '{name}' missing fundamentals data columns: {missing}; "
            "run data fetch for the required Tushare endpoints and build_data_table first"
        )
    return frame


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
    bars = load_daily_bars(
        lake,
        start=f"{load_start:%Y%m%d}",
        end=f"{load_end:%Y%m%d}",
        symbols=symbols,
    )
    if bars.empty:
        raise ValueError(
            "no daily bars found in data lake; run data fetch and build_data_table first"
        )
    factor_frame = compute_factor_frame(bars, name, registry=factor_registry)
    validation = factor_frame.merge(
        _forward_returns(bars),
        on=["symbol", "trade_date"],
        how="inner",
    )
    validation = validation[
        (validation["trade_date"] >= start_date) & (validation["trade_date"] <= end_date)
    ].reset_index(drop=True)
    if validation.empty:
        raise ValueError(f"no validation rows between {start_date} and {end_date}")
    return validation


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
