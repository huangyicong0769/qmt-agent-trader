from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    _abrupt_low_coverage_dates,
)


def test_abrupt_cross_section_collapse_is_blocking() -> None:
    metadata = {
        "daily_cross_sectional_coverage": {"2024-03-15": 0.015},
        "daily_reference_symbol_counts": {"2024-03-15": 2_000.0},
    }
    config = StrategyBacktestConfig(
        strategy_id="strategy",
        start_date="20240101",
        end_date="20240630",
        top_n=20,
    )
    assert _abrupt_low_coverage_dates(metadata, config) == ["2024-03-15"]


def test_small_universe_does_not_trigger_broad_coverage_gate() -> None:
    metadata = {
        "daily_cross_sectional_coverage": {"2024-03-15": 0.10},
        "daily_reference_symbol_counts": {"2024-03-15": 20.0},
    }
    config = StrategyBacktestConfig(
        strategy_id="strategy",
        start_date="20240101",
        end_date="20240630",
        top_n=5,
    )
    assert _abrupt_low_coverage_dates(metadata, config) == []
