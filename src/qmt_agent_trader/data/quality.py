"""Lightweight data quality checks."""

from __future__ import annotations

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: set[str]) -> list[str]:
    return sorted(columns.difference(frame.columns))


def has_duplicate_keys(frame: pd.DataFrame, keys: list[str]) -> bool:
    return bool(frame.duplicated(subset=keys).any())
