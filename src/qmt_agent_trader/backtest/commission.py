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


def calculate_cost(notional: float, side: Side, config: CostConfig | None = None) -> float:
    cfg = config or CostConfig()
    commission = max(notional * cfg.commission_rate, cfg.min_commission)
    stamp_tax = notional * cfg.stamp_tax_rate if side == Side.SELL else 0.0
    transfer_fee = notional * cfg.transfer_fee_rate
    return commission + stamp_tax + transfer_fee
