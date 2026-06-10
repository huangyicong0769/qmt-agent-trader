"""Factor aggregation helpers."""

from __future__ import annotations

import pandas as pd


def weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        score = score + frame[column].rank(pct=True) * weight
    return score
