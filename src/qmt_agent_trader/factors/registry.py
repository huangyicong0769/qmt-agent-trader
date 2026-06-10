"""Factor registry."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

FactorFn = Callable[[pd.DataFrame], pd.Series]


class FactorRegistry:
    def __init__(self) -> None:
        self._factors: dict[str, FactorFn] = {}

    def register(self, name: str, fn: FactorFn) -> None:
        if name in self._factors:
            raise ValueError(f"factor already registered: {name}")
        self._factors[name] = fn

    def compute(self, name: str, frame: pd.DataFrame) -> pd.Series:
        return self._factors[name](frame)

    def names(self) -> list[str]:
        return sorted(self._factors)
