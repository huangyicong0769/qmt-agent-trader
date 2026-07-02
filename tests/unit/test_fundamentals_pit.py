from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.leakage_checks import assert_fundamentals_visible
from qmt_agent_trader.core.errors import LeakageError
from qmt_agent_trader.data.fundamentals import (
    load_daily_basic_snapshot,
    load_financials_asof,
    load_fundamentals_asof,
    normalize_financial_statement,
)
from qmt_agent_trader.data.storage import DataLake


def test_normalize_financial_statement_prefers_actual_announcement_date() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240120",
                "f_ann_date": "20240121",
                "roe": 0.11,
            }
        ]
    )

    result = normalize_financial_statement(
        frame,
        statement_type="fina_indicator",
        source="tushare/fina_indicator",
    )

    assert result["period_end"].tolist() == [date(2023, 12, 31)]
    assert result["announced_at"].tolist() == [date(2024, 1, 21)]
    assert result["visible_date"].tolist() == [date(2024, 1, 21)]
    assert result["pit_safe"].tolist() == [True]


def test_normalize_financial_statement_marks_missing_ann_date_not_pit_safe() -> None:
    frame = pd.DataFrame([{"ts_code": "000001.SZ", "end_date": "20231231", "roe": 0.11}])

    result = normalize_financial_statement(
        frame,
        statement_type="fina_indicator",
        source="tushare/fina_indicator",
    )

    assert result["pit_safe"].tolist() == [False]


def test_load_daily_basic_snapshot_uses_latest_trade_date_asof(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "pe_ttm": 5.0},
                {"ts_code": "000001.SZ", "trade_date": "20240201", "pe_ttm": 6.0},
            ]
        ),
        "raw",
        "tushare/daily_basic",
    )

    result = load_daily_basic_snapshot(lake, as_of_date="20240131", symbols=["000001.SZ"])

    assert result[["symbol", "trade_date", "pe_ttm"]].to_dict("records") == [
        {"symbol": "000001.SZ", "trade_date": date(2024, 1, 2), "pe_ttm": 5.0}
    ]


def test_load_financials_asof_filters_future_announcements(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20230930",
                    "ann_date": "20231025",
                    "roe": 0.10,
                    "gross_margin": 30.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240201",
                    "roe": 0.12,
                    "gross_margin": 32.0,
                },
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )

    result = load_financials_asof(
        lake,
        as_of_date="20240131",
        symbols=["000001.SZ"],
        fields=["roe", "gross_margin"],
    )

    assert result[["symbol", "roe", "gross_margin", "latest_period_end"]].to_dict(
        "records"
    ) == [
        {
            "symbol": "000001.SZ",
            "roe": 0.10,
            "gross_margin": 30.0,
            "latest_period_end": date(2023, 9, 30),
        }
    ]


def test_load_fundamentals_asof_merges_daily_and_financials(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240131",
                    "pe_ttm": 4.8,
                    "pb": 0.55,
                    "total_mv": 1000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily_basic",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20230930",
                    "ann_date": "20231025",
                    "roe": 0.11,
                    "gross_margin": 31.0,
                }
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )

    result = load_fundamentals_asof(
        lake,
        as_of_date="20240131",
        symbols=["000001.SZ"],
        fields=["pe_ttm", "pb", "roe", "gross_margin", "total_mv"],
    )

    assert result.iloc[0]["pe_ttm"] == 4.8
    assert result.iloc[0]["pb"] == 0.55
    assert result.iloc[0]["roe"] == 0.11
    assert result.iloc[0]["latest_announced_at"] == date(2023, 10, 25)


def test_assert_fundamentals_visible_rejects_future_rows() -> None:
    frame = pd.DataFrame([{"symbol": "000001.SZ", "visible_date": "2024-02-01"}])

    with pytest.raises(LeakageError, match="future visible_date"):
        assert_fundamentals_visible(frame, date(2024, 1, 31))
