from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.universe.pit_metadata import (
    index_interval_members_asof,
    index_weight_members_asof,
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
