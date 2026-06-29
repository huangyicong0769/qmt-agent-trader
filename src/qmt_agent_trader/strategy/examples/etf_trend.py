"""ETF trend-following template."""

from __future__ import annotations

import pandas as pd

from qmt_agent_trader.strategy.base import StrategyContext


class ETFTrendStrategy:
    strategy_id = "etf_trend_v1"
    version = "1.0.0"

    def __init__(
        self,
        *,
        short_ma: int = 20,
        long_ma: int = 60,
        max_single_position_pct: float = 1.0,
    ) -> None:
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.max_single_position_pct = max_single_position_pct

    def generate_signals(self, data: pd.DataFrame | StrategyContext) -> pd.DataFrame:
        frame = data.bars.copy() if isinstance(data, StrategyContext) else data.copy()
        if frame.empty:
            return _empty_signals()
        if not {"symbol", "trade_date", "close"}.issubset(frame.columns):
            raise ValueError("ETF trend strategy requires symbol, trade_date, close")
        frame = frame.sort_values(["symbol", "trade_date"]).copy()
        grouped = frame.groupby("symbol", group_keys=False)
        frame["short_ma"] = grouped["close"].transform(
            lambda values: values.rolling(self.short_ma, min_periods=1).mean()
        )
        frame["long_ma"] = grouped["close"].transform(
            lambda values: values.rolling(self.long_ma, min_periods=1).mean()
        )
        latest = frame.drop_duplicates("symbol", keep="last").copy()
        latest["score"] = latest["short_ma"] / latest["long_ma"] - 1.0
        risk_on = latest["score"] > 0
        if not risk_on.any():
            return _empty_signals()
        selected = latest.loc[risk_on].sort_values("score", ascending=False).copy()
        weight = min(1.0 / len(selected), self.max_single_position_pct)
        selected["target_weight"] = weight
        selected["reason"] = f"trend:{self.short_ma}>{self.long_ma}"
        selected["signal_date"] = selected["trade_date"].astype(str)
        return selected[["symbol", "signal_date", "score", "target_weight", "reason"]].reset_index(
            drop=True
        )


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "signal_date", "score", "target_weight", "reason"])
