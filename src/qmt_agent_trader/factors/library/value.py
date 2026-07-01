"""Value factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def size_log_mktcap(frame: pd.DataFrame) -> pd.Series:
    column = "total_mv" if "total_mv" in frame.columns else "circ_mv"
    return np.log(frame[column].clip(lower=1))


def pe_ttm_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["pe_ttm"].rank(pct=True, ascending=True)


def pb_rank(frame: pd.DataFrame) -> pd.Series:
    return frame["pb"].rank(pct=True, ascending=True)


def dividend_yield(frame: pd.DataFrame) -> pd.Series:
    column = "dv_ttm" if "dv_ttm" in frame.columns else "dividend_yield"
    return frame[column]
