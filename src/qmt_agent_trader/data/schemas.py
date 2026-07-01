"""Shared data query schema helpers."""

from __future__ import annotations

import pandas as pd

DATA_STATUS_OK = "OK"
DATA_STATUS_PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
DATA_STATUS_NO_DATA = "NO_DATA"
DATA_STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
DATA_STATUS_INVALID_REQUEST = "INVALID_REQUEST"
DATA_STATUS_PIT_NOT_VALIDATED = "PIT_NOT_VALIDATED"
DATA_STATUS_ERROR = "ERROR"


def select_existing_columns(
    frame: pd.DataFrame,
    requested: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    existing = [column for column in requested if column in frame.columns]
    missing = [column for column in requested if column not in frame.columns]
    return frame[existing].copy(), missing
