"""Account models."""

from __future__ import annotations

from pydantic import BaseModel


class AccountAsset(BaseModel):
    account_id_hash: str
    cash: float
    total_asset: float
    market_value: float


class Position(BaseModel):
    symbol: str
    quantity: int
    available_quantity: int
    market_value: float
    cost_price: float | None = None
