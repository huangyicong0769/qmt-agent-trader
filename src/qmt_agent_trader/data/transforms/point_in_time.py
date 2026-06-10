"""Point-in-time visibility helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd


def filter_visible_by_announcement(
    frame: pd.DataFrame, as_of: date, announcement_column: str = "announced_at"
) -> pd.DataFrame:
    data = frame.copy()
    data[announcement_column] = pd.to_datetime(data[announcement_column]).dt.date
    return data[data[announcement_column] <= as_of]


def assert_no_future_rows(frame: pd.DataFrame, as_of: date, date_column: str) -> None:
    visible_dates = pd.to_datetime(frame[date_column]).dt.date
    if bool((visible_dates > as_of).any()):
        raise ValueError(f"{date_column} contains data after {as_of}")
