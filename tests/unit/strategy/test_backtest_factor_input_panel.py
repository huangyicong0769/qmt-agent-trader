from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.models import FactorLeg, StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def test_backtest_pb_rank_uses_daily_basic_input_panel(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ"])
    _write_daily_basic(lake, field="pb", values={"000001.SZ": 0.8, "000002.SZ": 1.5})

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        _config("pb_rank", symbols=["000001.SZ", "000002.SZ"]),
        reports_dir=tmp_path / "reports",
    )

    assert result.status == "completed"
    assert result.requested_factor_ids == ["pb_rank"]
    assert result.factor_ids == ["pb_rank"]
    assert result.research_only is True
    assert result.live_trading_allowed is False


def test_backtest_dividend_yield_uses_daily_basic_input_panel(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ"])
    _write_daily_basic(lake, field="dv_ttm", values={"000001.SZ": 0.03, "000002.SZ": 0.01})

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        _config("dividend_yield", symbols=["000001.SZ", "000002.SZ"]),
        reports_dir=tmp_path / "reports",
    )

    assert result.status == "completed"
    assert result.requested_factor_ids == ["dividend_yield"]


def test_backtest_roe_rank_uses_pit_visible_fina_indicator(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    _write_bars(
        lake,
        start=date(2024, 1, 29),
        days=8,
        symbols=["000001.SZ", "000002.SZ"],
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20230930",
                    "ann_date": "20231025",
                    "roe": 0.10,
                },
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240201",
                    "roe": 0.20,
                },
                {
                    "ts_code": "000002.SZ",
                    "end_date": "20230930",
                    "ann_date": "20231025",
                    "roe": 0.08,
                },
                {
                    "ts_code": "000002.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240201",
                    "roe": 0.18,
                },
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        _config(
            "roe_rank",
            start_date="20240129",
            end_date="20240205",
            symbols=["000001.SZ", "000002.SZ"],
        ),
        reports_dir=tmp_path / "reports",
    )

    assert result.status == "completed"
    assert result.requested_factor_ids == ["roe_rank"]


def test_backtest_composite_strategy_unions_all_factor_required_columns(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ"])
    _write_daily_basic(lake, field="pb", values={"000001.SZ": 0.8, "000002.SZ": 1.5})
    spec = StrategySpec(
        strategy_id="value_momentum",
        name="Value momentum",
        factors=[
            FactorLeg(factor_id="pb_rank", weight=0.5),
            FactorLeg(factor_id="momentum_60d", weight=0.5),
        ],
        portfolio={"top_n": 1},
    )

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        StrategyBacktestConfig(
            strategy_id=spec.strategy_id,
            strategy_spec=spec,
            factor_name="pb_rank",
            start_date="20240101",
            end_date="20240320",
            symbols=["000001.SZ", "000002.SZ"],
            top_n=1,
        ),
        reports_dir=tmp_path / "reports",
    )

    assert result.status == "completed"
    assert result.requested_factor_ids == ["pb_rank", "momentum_60d"]
    assert result.factor_ids == ["pb_rank", "momentum_60d"]
    assert result.execution_backend == "factor_rank_composite_adapter"


def test_backtest_missing_pb_blocks_with_panel_repair_guidance(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ"])

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        _config("pb_rank", symbols=["000001.SZ", "000002.SZ"]),
        reports_dir=tmp_path / "reports",
    )

    assert result.status in {"BLOCKED", "DATA_NOT_READY"}
    assert result.reason in {"MISSING_FACTOR_INPUTS", "INPUT_PANEL_PARTIAL_COVERAGE"}
    assert result.missing_columns == ["pb"]
    assert result.next_repair_tool == "run_tushare_fetch"
    assert result.suggested_repair["items"][0]["api_name"] == "daily_basic"
    assert result.input_panel_metadata["missing_fields"]["pb"]["api_name"] == "daily_basic"


def test_backtest_blocks_required_factor_field_below_coverage_threshold(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ"])
    _write_daily_basic(
        lake,
        field="pb",
        values={"000001.SZ": 0.8},
        days=5,
    )

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        _config("pb_rank", symbols=["000001.SZ", "000002.SZ"]),
        reports_dir=tmp_path / "reports",
    )

    assert result.status == "BLOCKED"
    assert result.reason == "INPUT_PANEL_PARTIAL_COVERAGE"
    assert result.coverage_by_field["pb"]["coverage"] < 0.80
    assert result.input_panel_metadata["evidence_status"] == "BLOCKED"


def test_partial_coverage_warning_only_when_above_minimum(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ"])
    _write_daily_basic(
        lake,
        field="pb",
        values={"000001.SZ": 0.8, "000002.SZ": 1.5},
        days=70,
    )

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        _config("pb_rank", symbols=["000001.SZ", "000002.SZ"]),
        reports_dir=tmp_path / "reports",
    )

    assert result.status == "completed"
    assert result.coverage_by_field["pb"]["coverage"] >= 0.80
    assert result.coverage_by_field["pb"]["source"] == "raw/tushare/daily_basic"
    assert result.coverage_by_field["pb"]["join_policy"] == "exact"
    assert result.input_panel_metadata["evidence_status"] == "STRONG"
    assert any(warning.startswith("input_panel_partial_coverage:pb") for warning in result.warnings)


def test_backtest_technical_factor_still_runs_from_bars(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    _write_bars(lake, symbols=["000001.SZ", "000002.SZ"])

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        _config("momentum_20d", symbols=["000001.SZ", "000002.SZ"]),
        reports_dir=tmp_path / "reports",
    )

    assert result.status == "completed"
    assert result.requested_factor_ids == ["momentum_20d"]


def _lake(tmp_path: Path) -> DataLake:
    return DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")


def _config(
    factor_name: str,
    *,
    start_date: str = "20240101",
    end_date: str = "20240320",
    symbols: list[str],
) -> StrategyBacktestConfig:
    return StrategyBacktestConfig(
        strategy_id=f"factor_{factor_name}",
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        top_n=1,
    )


def _write_bars(
    lake: DataLake,
    *,
    start: date = date(2024, 1, 1),
    days: int = 80,
    symbols: list[str],
) -> None:
    rows: list[dict[str, object]] = []
    for offset in range(days):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        for symbol_index, symbol in enumerate(symbols):
            base = 10.0 + symbol_index * 4.0
            drift = offset * (0.10 + symbol_index * 0.03)
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "open": base + drift,
                    "high": base + drift + 0.5,
                    "low": base + drift - 0.5,
                    "close": base + drift + (0.1 if offset % 2 else -0.1),
                    "vol": 100000 + symbol_index * 1000 + offset,
                    "amount": 1000000 + symbol_index * 10000 + offset * 100,
                    "turnover": 0.02 + symbol_index * 0.01,
                }
            )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")


def _write_daily_basic(
    lake: DataLake,
    *,
    field: str,
    values: dict[str, float],
    start: date = date(2024, 1, 1),
    days: int = 80,
) -> None:
    rows: list[dict[str, object]] = []
    for offset in range(days):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        for symbol, value in values.items():
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    field: value + offset * 0.001,
                }
            )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily_basic")
