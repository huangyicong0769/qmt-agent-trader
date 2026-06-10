"""Factor validation guardrails."""

from __future__ import annotations

import pandas as pd


def assert_no_missing_factor_values(values: pd.Series) -> None:
    if bool(values.isna().any()):
        raise ValueError("factor contains missing values")
