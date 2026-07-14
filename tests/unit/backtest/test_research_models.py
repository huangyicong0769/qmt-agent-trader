from qmt_agent_trader.backtest.research_models import (
    FactorRankResearchResult,
    ResearchDataQuality,
    ResearchEquityPoint,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityMetrics


def test_result_serializes_dated_equity_and_legacy_curve() -> None:
    result = FactorRankResearchResult(
        metrics=SensitivityMetrics(total_return=0.1, turnover=0.2),
        trades=(),
        equity_points=(
            ResearchEquityPoint(
                trade_date="2024-01-02",
                cash=100.0,
                market_value=0.0,
                equity=100.0,
                stale_position_count=0,
            ),
            ResearchEquityPoint(
                trade_date="2024-01-03",
                cash=90.0,
                market_value=20.0,
                equity=110.0,
                stale_position_count=0,
            ),
        ),
        rebalance_points=(),
        data_quality=ResearchDataQuality(),
    )

    payload = result.as_dict()

    assert payload["equity_curve"] == [100.0, 110.0]
    assert payload["equity_points"][1]["trade_date"] == "2024-01-03"
    assert payload["data_quality"]["validated_valuation_dates"] == 0
    assert "missing_held_price_events" not in payload["data_quality"]
    assert "stale_valuation_dates" not in payload["data_quality"]


def test_signal_availability_counts_are_canonical_data_quality() -> None:
    result = FactorRankResearchResult(
        metrics=SensitivityMetrics(total_return=0.0),
        trades=(),
        equity_points=(),
        rebalance_points=(),
        data_quality=ResearchDataQuality(
            scheduled_rebalance_count=5,
            available_signal_count=3,
            signal_unavailable_count=2,
        ),
    )

    payload = result.as_dict()

    assert payload["data_quality"]["scheduled_rebalance_count"] == 5
    assert payload["data_quality"]["available_signal_count"] == 3
    assert payload["data_quality"]["signal_unavailable_count"] == 2
    assert payload["scheduled_rebalance_count"] == 5
