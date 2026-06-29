import pytest

from qmt_agent_trader.strategy.loader import (
    StrategyLoadError,
    load_strategy_from_file,
    static_check_strategy_file,
)


def test_loader_rejects_broker_import(tmp_path) -> None:
    path = tmp_path / "strategy.py"
    path.write_text("from qmt_agent_trader.broker.order import Order\n", encoding="utf-8")

    assert static_check_strategy_file(path)
    with pytest.raises(StrategyLoadError):
        load_strategy_from_file(path)


def test_loader_rejects_submit_order(tmp_path) -> None:
    path = tmp_path / "strategy.py"
    path.write_text("def submit_order():\n    pass\n", encoding="utf-8")

    assert static_check_strategy_file(path)


def test_loader_loads_valid_generated_strategy(tmp_path) -> None:
    path = tmp_path / "strategy.py"
    path.write_text(
        """
import pandas as pd

STRATEGY_SPEC = {"strategy_id": "strat_ok", "name": "ok", "factors": ["momentum_20d"]}

def generate_signals(context):
    return pd.DataFrame(columns=["symbol", "target_weight"])
""",
        encoding="utf-8",
    )

    loaded = load_strategy_from_file(path)

    assert loaded.strategy_id == "strat_ok"
