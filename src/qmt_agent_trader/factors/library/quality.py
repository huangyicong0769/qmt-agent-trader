"""Quality factor placeholders."""

from __future__ import annotations

import pandas as pd


def roe_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["roe"].rank(pct=True, ascending=False)
