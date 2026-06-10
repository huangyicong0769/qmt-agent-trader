"""Factor computation service backed by canonical daily bars."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.library.price_volume import (
    amount_zscore_20d,
    momentum,
    reversal_5d,
    turnover_20d,
    volatility_20d,
)


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


def compute_factor_frame(bars: pd.DataFrame, name: str) -> pd.DataFrame:
    data = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True).copy()
    if name == "momentum_20d":
        values = momentum(data, 20)
    elif name == "momentum_60d":
        values = momentum(data, 60)
    elif name == "reversal_5d":
        values = reversal_5d(data)
    elif name == "volatility_20d":
        values = volatility_20d(data)
    elif name == "turnover_20d":
        values = turnover_20d(data)
    elif name == "amount_zscore_20d":
        values = amount_zscore_20d(data)
    else:
        raise ValueError(f"unsupported factor for current data set: {name}")

    return pd.DataFrame(
        {
            "symbol": data["symbol"],
            "trade_date": data["trade_date"],
            "factor_name": name,
            "factor_value": values,
        }
    )


def compute_factor_to_lake(lake: DataLake, *, name: str, date: str) -> FactorComputeResult:
    target_date = pd.to_datetime(date).date()
    bars = load_daily_bars(lake, end=target_date)
    if bars.empty:
        raise ValueError("no daily bars found in data lake; run data update first")

    factor_frame = compute_factor_frame(bars, name)
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
) -> FactorValidationResult:
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    bars = load_daily_bars(lake)
    if bars.empty:
        raise ValueError("no daily bars found in data lake; run data update first")

    factor_frame = compute_factor_frame(bars, name)
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


def _forward_returns(bars: pd.DataFrame) -> pd.DataFrame:
    data = bars.sort_values(["symbol", "trade_date"]).copy()
    next_close = data.groupby("symbol")["close"].shift(-1)
    data["forward_return_1d"] = next_close / data["close"] - 1
    return data[["symbol", "trade_date", "forward_return_1d"]]
