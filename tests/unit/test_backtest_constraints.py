import pandas as pd

from qmt_agent_trader.backtest.commission import calculate_cost
from qmt_agent_trader.backtest.constraints import TradeState, is_tradeable
from qmt_agent_trader.backtest.engine import DailyBacktestEngine
from qmt_agent_trader.core.types import Side


def test_t_plus_one_execution_uses_next_trade_date() -> None:
    bars = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "trade_date": "2026-06-09", "open": 10.0},
            {"symbol": "000001.SZ", "trade_date": "2026-06-10", "open": 11.0},
        ]
    )
    result = DailyBacktestEngine().run_one_signal(
        bars,
        symbol="000001.SZ",
        signal_date="2026-06-09",
        side=Side.BUY,
        quantity=100,
    )
    assert len(result.fills) == 1
    assert str(result.fills[0].trade_date) == "2026-06-10"
    assert result.fills[0].price == 11.0


def test_suspended_and_limit_rules() -> None:
    assert not is_tradeable(Side.BUY, TradeState(suspended=True))
    assert not is_tradeable(Side.BUY, TradeState(limit_up=True))
    assert not is_tradeable(Side.SELL, TradeState(limit_down=True))
    assert is_tradeable(Side.BUY, TradeState())


def test_commission_and_stamp_tax() -> None:
    buy_cost = calculate_cost(10000, Side.BUY)
    sell_cost = calculate_cost(10000, Side.SELL)
    assert sell_cost > buy_cost
