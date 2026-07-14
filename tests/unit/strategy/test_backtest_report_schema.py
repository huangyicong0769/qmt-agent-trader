from qmt_agent_trader.strategy.execution_adapter import _canonical_result_evidence


def test_strategy_backtest_report_exposes_canonical_evidence() -> None:
    result = {
        "equity_points": [{"trade_date": "2024-01-02", "equity": 100_000.0}],
        "rebalance_points": [],
        "trades": [],
        "data_quality": {"validated_valuation_dates": 1},
        "total_explicit_cost": 12.0,
        "total_slippage_cost": 3.0,
    }
    metrics = {
        "total_return": -0.12,
        "net_total_return": -0.12,
        "same_trade_gross_return": -0.105,
        "cost_drag": 0.015,
    }
    evidence = _canonical_result_evidence(result, metrics)

    assert evidence["equity_points"][0]["trade_date"] == "2024-01-02"
    assert isinstance(evidence["rebalance_points"], list)
    assert isinstance(evidence["trade_blotter"], list)
    assert evidence["cost_attribution"]["explicit_cost"] == 12.0
    assert evidence["data_quality"]["validated_valuation_dates"] == 1
