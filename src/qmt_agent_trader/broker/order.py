"""Order models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from qmt_agent_trader.core.types import OrderType, Side


class Order(BaseModel):
    model_config = {"frozen": True}

    symbol: str
    side: Side
    quantity: int = Field(gt=0)
    order_type: OrderType
    limit_price: float | None = Field(default=None, gt=0)
    reason: str

    @model_validator(mode="after")
    def validate_limit_price(self) -> Order:
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for LIMIT orders")
        return self
