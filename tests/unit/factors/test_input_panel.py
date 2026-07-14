from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.input_panel import (
    _daily_panel_coverage,
    build_target_frequency_panel,
)


def _write_daily(
    lake: DataLake,
    *,
    start: date,
    days: int,
    symbols: list[str] | None = None,
) -> None:
    symbols = symbols or ["000001.SZ"]
    rows = []
    for offset in range(days):
        trade_date = start + timedelta(days=offset)
        for index, symbol in enumerate(symbols):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": f"{trade_date:%Y%m%d}",
                    "open": 10.0 + offset + index,
                    "high": 11.0 + offset + index,
                    "low": 9.0 + offset + index,
                    "close": 10.0 + offset + index,
                    "vol": 1000.0,
                    "amount": 10000.0,
                }
            )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": row["ts_code"],
                    "trade_date": row["trade_date"],
                    "up_limit": float(row["close"]) * 1.1,
                    "down_limit": float(row["close"]) * 0.9,
                }
                for row in rows
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )


def test_daily_exact_source_joins_only_on_matching_trade_date(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    _write_daily(lake, start=date(2024, 1, 1), days=3)
    lake.write_parquet(
        pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20240102", "dv_ttm": 0.03}]
        ),
        "raw",
        "tushare/daily_basic",
    )

    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start="20240101",
        target_end="20240103",
        required_fields=["close", "dv_ttm"],
    )

    assert metadata["field_sources"]["dv_ttm"]["fill_policy"] == "exact"
    values = panel.set_index("trade_date")["dv_ttm"]
    assert pd.isna(values.loc[date(2024, 1, 1)])
    assert values.loc[date(2024, 1, 2)] == 0.03
    assert pd.isna(values.loc[date(2024, 1, 3)])


def test_low_frequency_financial_field_uses_visible_date_asof(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    _write_daily(lake, start=date(2024, 1, 1), days=10)
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240105",
                    "debt_to_assets": 0.42,
                }
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )

    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start="20240101",
        target_end="20240110",
        required_fields=["debt_to_assets"],
    )

    assert metadata["field_sources"]["debt_to_assets"]["fill_policy"] == "asof_snapshot"
    values = panel.set_index("trade_date")["debt_to_assets"]
    assert values.loc[date(2024, 1, 1) : date(2024, 1, 4)].isna().all()
    assert values.loc[date(2024, 1, 5) : date(2024, 1, 10)].tolist() == [0.42] * 6


def test_symbol_asof_does_not_leak_between_symbols(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    _write_daily(
        lake,
        start=date(2024, 1, 1),
        days=10,
        symbols=["000001.SZ", "000002.SZ"],
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240105",
                    "debt_to_assets": 0.42,
                },
                {
                    "ts_code": "000002.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240107",
                    "debt_to_assets": 0.66,
                },
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )

    panel, _metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start="20240101",
        target_end="20240110",
        required_fields=["debt_to_assets"],
    )

    values = panel.set_index(["symbol", "trade_date"])["debt_to_assets"]
    assert values.loc[("000001.SZ", date(2024, 1, 6))] == 0.42
    assert pd.isna(values.loc[("000002.SZ", date(2024, 1, 6))])
    assert values.loc[("000002.SZ", date(2024, 1, 7))] == 0.66


def test_marketwide_macro_asof_joins_to_all_symbols_after_visibility(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    _write_daily(
        lake,
        start=date(2024, 2, 10),
        days=11,
        symbols=["000001.SZ", "000002.SZ"],
    )
    lake.write_parquet(
        pd.DataFrame([{"month": "202401", "nt_val": 101.0}]),
        "raw",
        "tushare/cn_cpi",
    )

    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start="20240210",
        target_end="20240220",
        required_fields=["nt_val"],
    )

    assert metadata["field_sources"]["nt_val"]["api_name"] == "cn_cpi"
    before = panel[panel["trade_date"] < date(2024, 2, 15)]
    after = panel[panel["trade_date"] >= date(2024, 2, 15)]
    assert before["nt_val"].isna().all()
    assert after["nt_val"].tolist() == [101.0] * len(after)


def test_event_field_is_reported_unresolved_and_not_forward_filled(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    _write_daily(lake, start=date(2024, 1, 1), days=10)
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240105",
                    "div_proc": "实施",
                    "cash_div": 0.1,
                }
            ]
        ),
        "raw",
        "tushare/dividend",
    )

    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start="20240101",
        target_end="20240110",
        required_fields=["cash_div"],
    )

    assert "cash_div" not in panel.columns
    assert metadata["unresolved_fields"] == [
        {
            "field": "cash_div",
            "api_name": "dividend",
            "status": "UNRESOLVED_FIELD",
            "reason": "event_field_requires_explicit_transform",
            "suggested_next_step": "implement event_to_state transform or event-window factor",
        }
    ]
def test_daily_panel_coverage_uses_prior_rolling_symbol_reference() -> None:
    rows = []
    for offset in range(6):
        count = 3 if offset == 5 else 100
        for symbol in range(count):
            rows.append(
                {
                    "trade_date": date(2024, 1, 1) + timedelta(days=offset),
                    "symbol": f"{symbol:06d}.SZ",
                }
            )
    metadata = _daily_panel_coverage(pd.DataFrame(rows))
    assert metadata["daily_symbol_counts"]["2024-01-06"] == 3
    assert metadata["daily_cross_sectional_coverage"]["2024-01-06"] == 0.03


def test_exact_factor_source_duplicates_fail_closed(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    _write_daily(lake, start=date(2024, 1, 2), days=1)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "pb": 1.0},
                {"ts_code": "000001.SZ", "trade_date": "20240102", "pb": 1.2},
            ]
        ),
        "raw",
        "tushare/daily_basic",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        build_target_frequency_panel(
            lake,
            target_frequency=Frequency.DAILY,
            target_start="20240102",
            target_end="20240102",
            required_fields=["symbol", "trade_date", "open", "close", "pb"],
            symbols=["000001.SZ"],
        )

    assert exc_info.value.code == "DUPLICATE_EXACT_FACTOR_INPUT"
