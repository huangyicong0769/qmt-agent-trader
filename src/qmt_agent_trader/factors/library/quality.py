"""Quality factor placeholders."""

from __future__ import annotations

import pandas as pd


def roe_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["roe"].rank(pct=True, ascending=False)


def gross_margin_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["gross_margin"].rank(pct=True, ascending=False)


def debt_to_assets_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["debt_to_assets"].rank(pct=True, ascending=True)
