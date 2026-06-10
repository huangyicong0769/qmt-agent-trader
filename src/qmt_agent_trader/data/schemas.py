"""Canonical data schemas."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class DailyBar(BaseModel):
    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    suspended: bool = False
    limit_up: bool = False
    limit_down: bool = False
    st: bool = False


class Instrument(BaseModel):
    symbol: str
    name: str
    asset_type: str = Field(pattern="^(stock|etf|index)$")
    list_date: date | None = None
    delist_date: date | None = None
    exchange: str | None = None


class CorporateAction(BaseModel):
    symbol: str
    effective_date: date
    announced_at: date | None = None
    adjust_factor: float
