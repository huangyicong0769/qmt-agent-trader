"""Strategy base classes."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    strategy_id: str
    version: str

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame: ...
