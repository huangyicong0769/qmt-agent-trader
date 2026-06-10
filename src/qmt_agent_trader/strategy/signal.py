"""Signal models."""

from __future__ import annotations

from pydantic import BaseModel


class Signal(BaseModel):
    symbol: str
    target_weight: float
    reason: str
