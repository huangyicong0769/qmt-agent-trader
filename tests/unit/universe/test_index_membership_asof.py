from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.universe.pit_metadata import (
    index_interval_members_asof,
    index_weight_members_asof,
    index_weight_members_by_code_asof,
)


def test_index_weight_uses_latest_snapshot_not_historical_union() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "OLD.SZ",
                "trade_date": "20240101",
            },
            {
                "index_code": "000300.SH",
                "con_code": "NEW.SZ",
                "trade_date": "20240201",
            },
        ]
    )

    observed = index_weight_members_asof(
        frame,
        ["000300.SH"],
        date(2024, 2, 15),
    )

    assert observed == ["NEW.SZ"]


def test_index_member_uses_effective_interval() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "OLD.SZ",
                "in_date": "20230101",
                "out_date": "20240131",
            },
            {
                "index_code": "000300.SH",
                "con_code": "NEW.SZ",
                "in_date": "20240201",
                "out_date": None,
            },
        ]
    )

    observed = index_interval_members_asof(
        frame,
        ["000300.SH"],
        date(2024, 2, 15),
    )

    assert observed == ["NEW.SZ"]


def test_non_empty_invalid_out_date_fails_closed() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "20200101",
                "out_date": "bad-date",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_interval_members_asof(
            frame,
            ["000300.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_SOURCE_INVALID"
    assert exc_info.value.field == "raw/tushare/index_member.out_date"


def test_non_empty_invalid_in_date_fails_closed() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "bad-date",
                "out_date": None,
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_interval_members_asof(
            frame,
            ["000300.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_SOURCE_INVALID"
    assert exc_info.value.field == "raw/tushare/index_member.in_date"


def test_index_weight_returns_members_grouped_by_requested_code() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "300_A.SZ",
                "trade_date": "20240201",
            },
            {
                "index_code": "000905.SH",
                "con_code": "905_A.SZ",
                "trade_date": "20240202",
            },
        ]
    )

    observed = index_weight_members_by_code_asof(
        frame,
        ["000300.SH", "000905.SH"],
        date(2024, 2, 15),
    )

    assert observed == {
        "000300.SH": ["300_A.SZ"],
        "000905.SH": ["905_A.SZ"],
    }
