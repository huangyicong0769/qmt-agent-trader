"""ETF trend-following template."""

from __future__ import annotations

import pandas as pd


class ETFTrendStrategy:
    strategy_id = "etf_trend_v1"
    version = "1.0.0"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        output = data.copy()
        output["target_weight"] = (output["close"] > output["ma"]).astype(float)
        return output[["symbol", "target_weight"]]
