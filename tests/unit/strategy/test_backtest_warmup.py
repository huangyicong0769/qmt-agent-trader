from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.trading_calendar import load_session_window
from qmt_agent_trader.strategy import execution_adapter
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.registry import StrategyRegistry


def test_session_window_resolves_prior_open_days(tmp_path) -> None:
    lake = _calendar_lake(tmp_path)

    window = load_session_window(
        lake,
        start="20240104",
        end="20240105",
        warmup_sessions=2,
        exchanges=("SSE",),
    )

    assert window.warmup_dates == (date(2024, 1, 2), date(2024, 1, 3))
    assert window.expected_dates == (date(2024, 1, 4), date(2024, 1, 5))
    assert window.panel_start == date(2024, 1, 2)


def test_insufficient_warmup_history_fails_closed(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240104", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_session_window(
            lake,
            start="20240104",
            end="20240105",
            warmup_sessions=2,
            exchanges=("SSE",),
        )

    assert exc_info.value.code == "INSUFFICIENT_FACTOR_WARMUP_HISTORY"
    assert exc_info.value.details["required_sessions"] == 2
    assert exc_info.value.details["available_sessions"] == 1


def test_adapter_loads_factor_lookback_before_requested_start(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    calendar_dates = pd.date_range("2024-01-02", "2024-01-10", freq="D")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": f"{day:%Y%m%d}",
                    "is_open": 1,
                }
                for day in calendar_dates
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    observed: dict[str, object] = {}

    def fake_panel(_lake, **kwargs):
        observed["target_start"] = kwargs["target_start"]
        rows = [
            {
                "symbol": "000001.SZ",
                "trade_date": day.date(),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100.0,
                "amount": 1_000.0,
                "turnover": 0.01,
                "suspended": False,
                "limit_up": False,
                "limit_down": False,
                "st": False,
            }
            for day in calendar_dates
        ]
        panel = pd.DataFrame(rows)
        return panel, {
            "status": "OK",
            "input_panel_status": "OK",
            "evidence_status": "STRONG",
            "required_fields": list(panel.columns),
            "coverage_by_field": {
                column: {"coverage": 1.0} for column in panel.columns
            },
            "missing_fields": {},
            "unresolved_fields": [],
            "warnings": [],
            "daily_symbol_counts": {
                f"{day:%Y-%m-%d}": 1 for day in calendar_dates
            },
            "daily_cross_sectional_coverage": {
                f"{day:%Y-%m-%d}": 1.0 for day in calendar_dates
            },
        }

    class FakeRunner:
        def __init__(self, bars, config):
            observed["runner_dates"] = config.expected_trade_dates
            observed["bar_dates"] = tuple(sorted(bars["trade_date"].unique()))
            self.factor_frame = pd.DataFrame()
            self.bars = bars

        def run(self, scenario):
            raise BacktestDataIntegrityError(
                code="fixture_stop",
                message="stop after warm-up assertions",
            )

    monkeypatch.setattr(execution_adapter, "build_target_frequency_panel", fake_panel)
    monkeypatch.setattr(execution_adapter, "FactorRankResearchRunner", FakeRunner)

    with pytest.raises(BacktestDataIntegrityError, match="stop after warm-up"):
        run_strategy_backtest(
            lake,
            StrategyRegistry(tmp_path / "strategies"),
            StrategyBacktestConfig(
                strategy_id="factor_reversal_5d",
                factor_name="reversal_5d",
                start_date="20240109",
                end_date="20240110",
                symbols=["000001.SZ"],
                top_n=1,
            ),
            reports_dir=tmp_path / "reports",
        )

    assert observed["target_start"] == "20240104"
    assert observed["runner_dates"] == (date(2024, 1, 9), date(2024, 1, 10))
    assert observed["bar_dates"] == tuple(day.date() for day in calendar_dates)


def _calendar_lake(tmp_path) -> DataLake:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240101", "is_open": 0},
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240104", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    return lake
