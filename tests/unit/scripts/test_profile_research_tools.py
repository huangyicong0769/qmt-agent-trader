import importlib.util
from pathlib import Path


def _load_profile_module():
    script_path = Path(__file__).parents[3] / "scripts" / "profile_research_tools.py"
    spec = importlib.util.spec_from_file_location(
        "profile_research_tools",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_backtest_config_uses_valid_adhoc_identity() -> None:
    config = _load_profile_module()._profile_backtest_config(
        start="20240101",
        end="20240201",
        symbols=["000001.SZ", "000002.SZ"],
    )

    assert config.strategy_identity_mode == "adhoc"
    assert config.strategy_spec is not None
    assert config.strategy_id == config.strategy_spec.strategy_id
    assert config.factor_name == "momentum_20d"
    assert config.strategy_spec.factors[0].factor_id == "momentum_20d"
    assert config.top_n == 2
