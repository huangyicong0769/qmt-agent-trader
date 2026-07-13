from qmt_agent_trader.backtest.commission import (
    CostConfig,
    calculate_cost,
    calculate_cost_breakdown,
)
from qmt_agent_trader.core.types import Side


def test_cost_breakdown_matches_legacy_total() -> None:
    config = CostConfig()
    breakdown = calculate_cost_breakdown(100_000.0, Side.SELL, config)

    assert breakdown.commission == 30.0
    assert breakdown.stamp_tax == 50.0
    assert breakdown.transfer_fee == 1.0
    assert breakdown.total == calculate_cost(100_000.0, Side.SELL, config)


def test_zero_multiplier_cost_config_can_reach_zero() -> None:
    config = CostConfig(
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
    )
    assert calculate_cost_breakdown(100_000.0, Side.BUY, config).total == 0.0
