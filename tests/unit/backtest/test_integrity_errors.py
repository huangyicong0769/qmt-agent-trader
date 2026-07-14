from qmt_agent_trader.backtest.errors import BacktestAccountingError


def test_integrity_error_serializes_structured_context() -> None:
    error = BacktestAccountingError(
        code="NEGATIVE_CASH_AFTER_BUY",
        message="post-trade cash violated the non-negative invariant",
        trade_date="2024-01-03",
        symbols=("000001.SZ",),
        field="cash",
        details={"cash": -5.0, "tolerance": 1e-8},
    )

    assert error.as_dict() == {
        "code": "NEGATIVE_CASH_AFTER_BUY",
        "message": "post-trade cash violated the non-negative invariant",
        "trade_date": "2024-01-03",
        "symbols": ["000001.SZ"],
        "field": "cash",
        "details": {"cash": -5.0, "tolerance": 1e-8},
    }
