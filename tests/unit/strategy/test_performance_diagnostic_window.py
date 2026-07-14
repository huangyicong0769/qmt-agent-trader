from datetime import date

import pandas as pd

from qmt_agent_trader.strategy.execution_adapter import (
    _diagnostic_evidence,
    _performance_window,
)


def test_performance_window_excludes_warmup_dates() -> None:
    frame = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": date(2024, 1, 2), "factor_value": 100.0},
            {"symbol": "A", "trade_date": date(2024, 1, 4), "factor_value": 1.0},
            {"symbol": "A", "trade_date": date(2024, 1, 5), "factor_value": 2.0},
        ]
    )

    result = _performance_window(
        frame,
        (date(2024, 1, 4), date(2024, 1, 5)),
    )

    assert set(result["trade_date"]) == {date(2024, 1, 4), date(2024, 1, 5)}


def test_factor_diagnostic_ic_contains_no_warmup_date() -> None:
    rows = [
        ("A", date(2024, 1, 2), 100.0, 10.0),
        ("B", date(2024, 1, 2), -100.0, 10.0),
        ("A", date(2024, 1, 4), 2.0, 10.0),
        ("B", date(2024, 1, 4), 1.0, 10.0),
        ("A", date(2024, 1, 5), 2.0, 12.0),
        ("B", date(2024, 1, 5), 1.0, 9.0),
    ]
    factor_frame = pd.DataFrame(
        [
            {"symbol": symbol, "trade_date": trade_date, "factor_value": factor}
            for symbol, trade_date, factor, _close in rows
        ]
    )
    bars = pd.DataFrame(
        [
            {"symbol": symbol, "trade_date": trade_date, "close": close}
            for symbol, trade_date, _factor, close in rows
        ]
    )
    expected_dates = (date(2024, 1, 4), date(2024, 1, 5))

    evidence = _diagnostic_evidence(
        {"trades": [], "rejected_orders": 0, "data_quality": {}},
        {"valid": True},
        canonical_metrics={
            "average_top_n_overlap": None,
            "max_drawdown": 0.0,
            "average_one_way_turnover": 0.0,
            "explicit_cost_to_initial_cash": 0.0,
            "slippage_cost_to_initial_cash": 0.0,
            "cost_drag": 0.0,
        },
        factor_frame=_performance_window(factor_frame, expected_dates),
        bars=_performance_window(bars, expected_dates),
        initial_cash=1_000_000.0,
    )

    assert set(evidence["factor_report"]["ic_by_date"]) == {"20240104"}
