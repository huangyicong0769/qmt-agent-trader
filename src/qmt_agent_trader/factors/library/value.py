"""Value factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def size_log_mktcap(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame["mkt_cap"].clip(lower=1))


def pe_ttm_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["pe_ttm"].rank(pct=True, ascending=True)


def pb_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["pb"].rank(pct=True, ascending=True)


def dividend_yield(frame: pd.DataFrame) -> pd.Series:
    return frame["dividend_yield"]
