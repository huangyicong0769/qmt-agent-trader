"""Gateway schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    dry_run: bool
    live_trading_enabled: bool
    allow_order_endpoint: bool


class OrderPrecheckResponse(BaseModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


class SubmitPlanResponse(BaseModel):
    accepted: bool
    dry_run: bool
    idempotency_key: str | None = None
    execution_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
