"""Trading cost model."""

from __future__ import annotations

from dataclasses import dataclass

from qmt_agent_trader.core.types import Side


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    min_commission: float = 5.0


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    stamp_tax: float
    transfer_fee: float

    @property
    def total(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee


def calculate_cost_breakdown(
    notional: float,
    side: Side,
    config: CostConfig | None = None,
) -> CostBreakdown:
    cfg = config or CostConfig()
    commission = round(max(notional * cfg.commission_rate, cfg.min_commission), 10)
    stamp_tax = round(notional * cfg.stamp_tax_rate, 10) if side == Side.SELL else 0.0
    transfer_fee = round(notional * cfg.transfer_fee_rate, 10)
    return CostBreakdown(
        commission=commission,
        stamp_tax=stamp_tax,
        transfer_fee=transfer_fee,
    )


def calculate_cost(notional: float, side: Side, config: CostConfig | None = None) -> float:
    return calculate_cost_breakdown(notional, side, config).total
