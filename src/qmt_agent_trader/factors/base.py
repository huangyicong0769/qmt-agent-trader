"""Factor protocol."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Factor(Protocol):
    name: str

    def compute(self, bars: pd.DataFrame) -> pd.Series: ...
