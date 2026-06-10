"""Universe filters."""

from __future__ import annotations

import pandas as pd


def filter_tradeable_universe(frame: pd.DataFrame) -> pd.DataFrame:
    mask = ~frame.get("st", False) & ~frame.get("suspended", False)
    return frame[mask].copy()
