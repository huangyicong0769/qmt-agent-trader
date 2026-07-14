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


def test_canonical_data_quality_exposes_signal_counts() -> None:
    result = {
        "data_quality": {
            "validated_valuation_dates": 4,
            "scheduled_rebalance_count": 3,
            "available_signal_count": 2,
            "signal_unavailable_count": 1,
        },
        "equity_points": [],
        "rebalance_points": [],
        "trades": [],
    }
    evidence = _canonical_result_evidence(result, {})

    assert evidence["data_quality"]["scheduled_rebalance_count"] == 3
    assert evidence["data_quality"]["available_signal_count"] == 2
    assert evidence["data_quality"]["signal_unavailable_count"] == 1
