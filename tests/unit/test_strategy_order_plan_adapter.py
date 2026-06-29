import pandas as pd

from qmt_agent_trader.core.types import Side
from qmt_agent_trader.strategy.order_plan_adapter import build_order_plan_from_target_portfolio
from qmt_agent_trader.strategy.signal import TargetPortfolio, TargetPosition


def test_build_order_plan_from_target_portfolio_generates_dry_run_orders() -> None:
    plan = build_order_plan_from_target_portfolio(
        portfolio=TargetPortfolio(
            strategy_id="strat_1",
            as_of_date="20240102",
            positions=[
                TargetPosition(
                    symbol="000001.SZ",
                    target_weight=0.2,
                    reason="target",
                )
            ],
        ),
        current_positions={"000001.SZ": 100},
        prices=pd.DataFrame({"symbol": ["000001.SZ"], "price": [10.0]}),
        strategy_version="0.1.0",
        account_id_hash="paper",
        dry_run=True,
        equity=100_000,
    )

    assert plan.dry_run is True
    assert plan.orders[0].side == Side.BUY
    assert plan.orders[0].quantity == 1900
