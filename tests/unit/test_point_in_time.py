from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.leakage_checks import (
    assert_financials_visible,
    assert_signal_before_execution,
)
from qmt_agent_trader.core.errors import LeakageError
from qmt_agent_trader.data.transforms.point_in_time import filter_visible_by_announcement


def test_financials_point_in_time_filter() -> None:
    frame = pd.DataFrame(
        [
            {"announced_at": "2026-06-01", "value": 1},
            {"announced_at": "2026-06-11", "value": 2},
        ]
    )
    visible = filter_visible_by_announcement(frame, date(2026, 6, 10))
    assert visible["value"].tolist() == [1]


def test_leakage_checks_fail_on_future_financials() -> None:
    frame = pd.DataFrame([{"announced_at": "2026-06-11", "value": 2}])
    with pytest.raises(LeakageError):
        assert_financials_visible(frame, date(2026, 6, 10))
    with pytest.raises(LeakageError):
        assert_signal_before_execution(date(2026, 6, 10), date(2026, 6, 10))
