"""Strategy base classes."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
from pydantic import BaseModel, Field


class StrategyContext(BaseModel):
    """Inputs injected by the runtime when executing a strategy."""

    as_of_date: str
    universe: str
    bars: Any
    factors: Any | None = None
    positions: dict[str, int] = Field(default_factory=dict)
    cash: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class Strategy(Protocol):
    strategy_id: str
    version: str

    def generate_signals(self, data: pd.DataFrame | StrategyContext) -> pd.DataFrame: ...
