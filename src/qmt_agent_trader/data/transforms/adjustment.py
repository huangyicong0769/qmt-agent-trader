"""Price adjustment helpers."""

from __future__ import annotations

import pandas as pd


def apply_visible_adjust_factor(
    frame: pd.DataFrame, price_columns: list[str], factor_column: str
) -> pd.DataFrame:
    adjusted = frame.copy()
    for column in price_columns:
        adjusted[column] = adjusted[column] * adjusted[factor_column]
    return adjusted
