import pandas as pd

from qmt_agent_trader.strategy.validation import validate_signals


def test_signal_validation_finds_duplicate_symbol() -> None:
    issues = validate_signals(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000001.SZ"],
                "target_weight": [0.1, 0.2],
            }
        )
    )

    assert any("duplicate" in issue for issue in issues)


def test_signal_validation_finds_weight_cap_breach() -> None:
    issues = validate_signals(
        pd.DataFrame({"symbol": ["000001.SZ"], "target_weight": [0.5]}),
        max_single_position_pct=0.1,
    )

    assert any("max_single_position" in issue for issue in issues)


def test_long_only_rejects_negative_weight() -> None:
    issues = validate_signals(
        pd.DataFrame({"symbol": ["000001.SZ"], "target_weight": [-0.1]}),
        long_only=True,
    )

    assert any("negative" in issue for issue in issues)
