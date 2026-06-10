"""A-share and ETF trading constraints."""

from __future__ import annotations

from dataclasses import dataclass

from qmt_agent_trader.core.types import Side


@dataclass(frozen=True)
class TradeState:
    suspended: bool = False
    limit_up: bool = False
    limit_down: bool = False
    st: bool = False
    delisting: bool = False
    listing_days: int | None = None


@dataclass(frozen=True)
class ConstraintConfig:
    filter_st: bool = True
    filter_delisting: bool = True
    min_listing_days: int = 60


def is_tradeable(side: Side, state: TradeState, config: ConstraintConfig | None = None) -> bool:
    cfg = config or ConstraintConfig()
    if state.suspended:
        return False
    if cfg.filter_st and state.st:
        return False
    if cfg.filter_delisting and state.delisting:
        return False
    if state.listing_days is not None and state.listing_days < cfg.min_listing_days:
        return False
    if side == Side.BUY and state.limit_up:
        return False
    if side == Side.SELL and state.limit_down:
        return False
    return True


def enforce_lot_size(quantity: int, lot_size: int = 100) -> int:
    return quantity // lot_size * lot_size
