import pandas as pd

from qmt_agent_trader.backtest.service import (
    compare_backtest_reports,
    run_backtest_report,
    run_single_symbol_backtest,
)
from qmt_agent_trader.data.storage import DataLake


def test_run_single_symbol_backtest_uses_next_day_fill(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240103",
                    "open": 11.0,
                    "high": 11.0,
                    "low": 11.0,
                    "close": 11.0,
                },
            ]
        ),
        "raw",
        "tushare/daily",
    )

    summary = run_single_symbol_backtest(
        lake,
        symbol="000001.SZ",
        signal_date="20240102",
        quantity=100,
    )

    assert summary.fills == 1
    assert summary.execution_dates == ["2024-01-03"]


def test_run_backtest_report_persists_report(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240103",
                    "open": 11.0,
                    "high": 11.0,
                    "low": 11.0,
                    "close": 11.0,
                },
            ]
        ),
        "raw",
        "tushare/daily",
    )
    reports_dir = tmp_path / "reports"

    summary = run_backtest_report(
        lake,
        reports_dir=reports_dir,
        symbol="000001.SZ",
        signal_date="20240102",
    )
    compared = compare_backtest_reports(reports_dir, limit=1)

    assert summary.report_path is not None
    assert compared["status"] == "compared"
    assert len(compared["runs"]) == 1
    report = compared["runs"][0]
    assert report["diagnostic_report"]["status"] == "WARN"
    check_names = {check["name"] for check in report["diagnostic_report"]["checks"]}
    assert {"leakage_valid", "min_observations", "min_trade_count"} <= check_names
