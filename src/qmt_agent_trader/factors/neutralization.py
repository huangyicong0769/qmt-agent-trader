"""Neutralization placeholder."""

from __future__ import annotations

import pandas as pd


def demean_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values - values.groupby(groups).transform("mean")
