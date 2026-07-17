from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.input_panel import build_target_frequency_panel


def _write_sources(lake: DataLake) -> None:
    daily_rows: list[dict[str, object]] = []
    basic_rows: list[dict[str, object]] = []
    limit_rows: list[dict[str, object]] = []
    for offset in range(25):
        day = date(2024, 1, 1) + timedelta(days=offset)
        key = f"{day:%Y%m%d}"
        daily_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": key,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "vol": 1000.0,
                "amount": 10000.0,
            }
        )
        basic_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": key,
                "turnover_rate": float(offset + 1),
            }
        )
        limit_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": key,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        )
    lake.write_parquet(pd.DataFrame(daily_rows), "raw", "tushare/daily")
    lake.write_parquet(pd.DataFrame(basic_rows), "raw", "tushare/daily_basic")
    lake.write_parquet(pd.DataFrame(limit_rows), "raw", "tushare/stk_limit")
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )


def test_canonical_turnover_comes_from_daily_basic_turnover_rate(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    _write_sources(lake)

    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start="20240101",
        target_end="20240125",
        required_fields=["turnover"],
        symbols=["000001.SZ"],
    )

    assert metadata["field_sources"]["turnover"]["api_name"] == "daily_basic"
    assert metadata["field_sources"]["turnover"]["source_field"] == "turnover_rate"
    assert panel["turnover"].tolist() == [float(index) for index in range(1, 26)]
