from datetime import date

import pytest

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.universe.timeline import RollingUniverseTimeline


def test_membership_uses_latest_snapshot_on_or_before_signal_date() -> None:
    timeline = RollingUniverseTimeline.from_mapping(
        {
            "20240105": ["A", "B"],
            "20240112": ["B", "C"],
        }
    )

    assert timeline.membership_as_of(date(2024, 1, 8)) == ("A", "B")
    assert timeline.membership_as_of(date(2024, 1, 12)) == ("B", "C")


def test_membership_before_first_snapshot_raises() -> None:
    timeline = RollingUniverseTimeline.from_mapping({"20240105": ["A"]})

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        timeline.membership_as_of(date(2024, 1, 4))

    assert exc_info.value.code == "ROLLING_UNIVERSE_SNAPSHOT_NOT_AVAILABLE"
