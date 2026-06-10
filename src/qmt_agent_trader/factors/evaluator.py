"""Factor validation metrics."""

from __future__ import annotations

import pandas as pd


def information_coefficient(factor: pd.Series, forward_return: pd.Series) -> float:
    return float(factor.corr(forward_return, method="spearman"))
