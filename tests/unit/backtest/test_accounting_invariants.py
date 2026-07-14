from datetime import date

import pytest

from qmt_agent_trader.backtest.commission import CostConfig
from qmt_agent_trader.backtest.errors import BacktestAccountingError
from qmt_agent_trader.backtest.research_runner import (
    _assert_equity_invariant,
    _assert_ledger_invariants,
    _max_affordable_buy_quantity,
)


def test_buy_affordability_includes_minimum_commission() -> None:
    quantity = _max_affordable_buy_quantity(
        cash=1_000.0,
        price=10.0,
        desired_quantity=100,
        cost_config=CostConfig(min_commission=5.0),
    )

    assert quantity == 0


def test_post_trade_negative_cash_raises_accounting_error() -> None:
    with pytest.raises(BacktestAccountingError) as exc_info:
        _assert_ledger_invariants(
            cash=-0.01,
            positions={"000001.SZ": 100},
            trade_date=date(2024, 1, 3),
        )

    assert exc_info.value.code == "NEGATIVE_CASH_AFTER_TRADE"


def test_non_finite_cash_raises_accounting_error() -> None:
    with pytest.raises(BacktestAccountingError) as exc_info:
        _assert_ledger_invariants(
            cash=float("nan"),
            positions={},
            trade_date=date(2024, 1, 3),
        )

    assert exc_info.value.code == "NON_FINITE_CASH"


def test_non_finite_equity_raises_accounting_error() -> None:
    with pytest.raises(BacktestAccountingError) as exc_info:
        _assert_equity_invariant(
            equity=float("inf"),
            trade_date=date(2024, 1, 3),
        )

    assert exc_info.value.code == "INVALID_EQUITY_VALUE"
