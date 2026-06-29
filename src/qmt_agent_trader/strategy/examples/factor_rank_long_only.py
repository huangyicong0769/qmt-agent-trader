"""Factor-rank long-only template."""

from __future__ import annotations

import pandas as pd

from qmt_agent_trader.strategy.base import StrategyContext
from qmt_agent_trader.strategy.models import FactorLeg
from qmt_agent_trader.strategy.portfolio import equal_weight_top_n_from_scores


class FactorRankLongOnlyStrategy:
    strategy_id = "factor_rank_long_only_v1"
    version = "1.0.0"

    def __init__(
        self,
        *,
        factors: list[FactorLeg | str] | None = None,
        top_n: int = 20,
        max_single_position_pct: float = 0.10,
        cash_buffer_pct: float = 0.02,
    ) -> None:
        self.factors = [
            item if isinstance(item, FactorLeg) else FactorLeg(factor_id=item)
            for item in (factors or [FactorLeg(factor_id="score")])
        ]
        self.top_n = top_n
        self.max_single_position_pct = max_single_position_pct
        self.cash_buffer_pct = cash_buffer_pct

    def generate_signals(self, data: pd.DataFrame | StrategyContext) -> pd.DataFrame:
        frame = _factor_frame(data)
        if frame.empty:
            return _empty_signals()
        scored = _score_factors(frame, self.factors)
        if scored.empty:
            return _empty_signals()
        signals = equal_weight_top_n_from_scores(
            scored,
            top_n=self.top_n,
            max_single_position_pct=self.max_single_position_pct,
            cash_buffer_pct=self.cash_buffer_pct,
            score_column="score",
        )
        trade_date = _latest_trade_date(scored)
        signals.insert(1, "signal_date", trade_date)
        signals.insert(2, "score", signals["symbol"].map(scored.set_index("symbol")["score"]))
        return signals[["symbol", "signal_date", "score", "target_weight", "reason"]]


def _factor_frame(data: pd.DataFrame | StrategyContext) -> pd.DataFrame:
    if isinstance(data, StrategyContext):
        if isinstance(data.factors, pd.DataFrame) and not data.factors.empty:
            return data.factors.copy()
        return data.bars.copy()
    return data.copy()


def _score_factors(frame: pd.DataFrame, factors: list[FactorLeg]) -> pd.DataFrame:
    if {"factor_id", "factor_value"}.issubset(frame.columns):
        wide = (
            frame.pivot_table(
                index=["symbol", "trade_date"] if "trade_date" in frame.columns else ["symbol"],
                columns="factor_id",
                values="factor_value",
                aggfunc="last",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    else:
        wide = frame.copy()
    if "symbol" not in wide.columns:
        raise ValueError("factor frame must include symbol")
    scored = wide.drop_duplicates("symbol", keep="last").copy()
    total_weight = sum(abs(leg.weight) for leg in factors if leg.factor_id in scored.columns)
    if total_weight <= 0 and "score" not in scored.columns:
        return pd.DataFrame(columns=["symbol", "trade_date", "score"])
    if total_weight <= 0:
        scored["score"] = pd.to_numeric(scored["score"], errors="coerce")
        return scored.dropna(subset=["score"])
    combined = pd.Series(0.0, index=scored.index)
    for leg in factors:
        if leg.factor_id not in scored.columns:
            continue
        values = pd.to_numeric(scored[leg.factor_id], errors="coerce")
        values = _normalize(values)
        if leg.ascending:
            values = -values
        combined = combined + values.fillna(0.0) * leg.weight
    scored["score"] = combined
    return scored.dropna(subset=["score"])


def _normalize(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std <= 0:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def _latest_trade_date(frame: pd.DataFrame) -> str:
    if "trade_date" not in frame.columns or frame["trade_date"].empty:
        return ""
    return str(frame["trade_date"].max())


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "signal_date", "score", "target_weight", "reason"])
