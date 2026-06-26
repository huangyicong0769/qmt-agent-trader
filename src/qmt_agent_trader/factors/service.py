"""Factor computation service backed by canonical daily bars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import pandas as pd

from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
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
    observations: int
    non_null: int
    coverage: float
    ic_mean: float | None
    ic_by_date: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "validated",
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "observations": self.observations,
            "non_null": self.non_null,
            "coverage": self.coverage,
            "ic_mean": self.ic_mean,
            "ic_by_date": self.ic_by_date,
        }


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


def compute_factor_frame(
    bars: pd.DataFrame,
    name: str,
    *,
    registry: FactorRegistry | None = None,
) -> pd.DataFrame:
    data = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True).copy()
    factor_registry = registry or FactorRegistry()
    values = factor_registry.compute(name, data)

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
    bars = load_daily_bars(lake, end=target_date)
    if bars.empty:
        raise ValueError("no daily bars found in data lake; run data update first")

    registry = FactorRegistry(Path(registry_root)) if registry_root is not None else None
    factor_frame = compute_factor_frame(bars, name, registry=registry)
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


def validate_factor(
    lake: DataLake,
    *,
    name: str,
    start: str,
    end: str,
    registry_root: str | None = None,
) -> FactorValidationResult:
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    bars = load_daily_bars(lake)
    if bars.empty:
        raise ValueError("no daily bars found in data lake; run data update first")

    registry = FactorRegistry(Path(registry_root)) if registry_root is not None else None
    factor_frame = compute_factor_frame(bars, name, registry=registry)
    returns = _forward_returns(bars)
    validation = factor_frame.merge(returns, on=["symbol", "trade_date"], how="inner")
    validation = validation[
        (validation["trade_date"] >= start_date) & (validation["trade_date"] <= end_date)
    ]
    if validation.empty:
        raise ValueError(f"no validation rows between {start_date} and {end_date}")

    usable = validation.dropna(subset=["factor_value", "forward_return_1d"])
    ic_by_date: dict[str, float] = {}
    for trade_date, group in usable.groupby("trade_date"):
        if len(group) < 2:
            continue
        ic = group["factor_value"].corr(group["forward_return_1d"], method="spearman")
        if pd.notna(ic):
            ic_by_date[f"{trade_date:%Y%m%d}"] = float(ic)

    ic_mean = float(pd.Series(ic_by_date).mean()) if ic_by_date else None
    observations = len(validation)
    non_null = len(usable)
    return FactorValidationResult(
        name=name,
        start=f"{start_date:%Y%m%d}",
        end=f"{end_date:%Y%m%d}",
        observations=observations,
        non_null=non_null,
        coverage=non_null / observations if observations else 0.0,
        ic_mean=ic_mean,
        ic_by_date=ic_by_date,
    )


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
) -> FactorWalkForwardResult:
    if window_days <= 1:
        raise ValueError("window_days must be greater than 1")
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if quantile <= 0 or quantile > 0.5:
        raise ValueError("quantile must be in (0, 0.5]")

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    bars = load_daily_bars(lake)
    if bars.empty:
        raise ValueError("no daily bars found in data lake; run data update first")

    registry = FactorRegistry(Path(registry_root)) if registry_root is not None else None
    factor_frame = compute_factor_frame(bars, name, registry=registry)
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


def _positive_slice(item: FactorWalkForwardSlice) -> bool:
    return (
        item.mean_ic is not None
        and item.long_short_spread is not None
        and item.mean_ic > 0
        and item.long_short_spread > 0
    )
