import pandas as pd

from qmt_agent_trader.strategy.portfolio import (
    equal_weight_top_n_from_scores,
    target_weights_to_quantities,
)


def test_equal_weight_top_n_applies_cash_buffer_and_cap() -> None:
    result = equal_weight_top_n_from_scores(
        pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "score": [3.0, 2.0, 1.0],
            }
        ),
        top_n=2,
        max_single_position_pct=0.4,
        cash_buffer_pct=0.1,
    )

    assert result["symbol"].tolist() == ["A", "B"]
    assert result["target_weight"].tolist() == [0.4, 0.4]


def test_target_weights_to_quantities_rounds_lot() -> None:
    result = target_weights_to_quantities(
        pd.DataFrame({"symbol": ["000001.SZ"], "target_weight": [0.123]}),
        equity=100_000,
        prices=pd.DataFrame({"symbol": ["000001.SZ"], "price": [10.0]}),
    )

    assert result.loc[0, "target_quantity"] == 1200
