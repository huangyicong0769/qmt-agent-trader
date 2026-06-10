"""Bar transforms."""

from __future__ import annotations

import pandas as pd


def sort_daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
