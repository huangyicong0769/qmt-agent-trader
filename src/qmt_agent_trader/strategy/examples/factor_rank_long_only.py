"""Factor-rank long-only template."""

from __future__ import annotations

import pandas as pd


class FactorRankLongOnlyStrategy:
    strategy_id = "factor_rank_long_only_v1"
    version = "1.0.0"

    def __init__(self, top_n: int = 20) -> None:
        self.top_n = top_n

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ranked = data.sort_values("score", ascending=False).head(self.top_n).copy()
        ranked["target_weight"] = 1 / len(ranked) if len(ranked) else 0
        return ranked[["symbol", "target_weight"]]
