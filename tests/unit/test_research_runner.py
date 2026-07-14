from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)
from qmt_agent_trader.backtest.sensitivity import SensitivityAnalyzer, SensitivityGrid
from qmt_agent_trader.factors.registry import FactorRegistry


def _bars() -> pd.DataFrame:
    start = date(2024, 1, 1)
    rows = []
    for offset in range(24):
        trade_date = start + timedelta(days=offset)
        rows.append(
            {
                "symbol": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + offset,
                "high": 10.5 + offset,
                "low": 9.5 + offset,
                "close": 10.0 + offset,
                "volume": 100000,
                "amount": 1000000,
            }
        )
        rows.append(
            {
                "symbol": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + offset * 0.1,
                "high": 20.5 + offset * 0.1,
                "low": 19.5 + offset * 0.1,
                "close": 20.0 + offset * 0.1,
                "volume": 100000,
                "amount": 1000000,
            }
        )
    return pd.DataFrame(rows)


def test_factor_rank_research_runner_executes_scenario() -> None:
    runner = FactorRankResearchRunner(
        _bars(),
        FactorRankResearchConfig(
            factor_name="momentum_20d",
            expected_trade_dates=tuple(sorted(_bars()["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=0.5,
            initial_cash=100000,
        ),
    )

    result = runner.run(SensitivityGrid().scenarios()[0])

    assert result.trades
    assert result.metrics.total_return > 0
    assert result.metrics.turnover > 0
    assert result.rejected_orders == 0


def test_sensitivity_analyzer_can_use_factor_rank_runner() -> None:
    runner = FactorRankResearchRunner(
        _bars(),
        FactorRankResearchConfig(
            factor_name="momentum_20d",
            expected_trade_dates=tuple(sorted(_bars()["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=0.5,
            initial_cash=100000,
        ),
    )
    grid = SensitivityGrid(
        cost_multipliers=(1.0, 2.0),
        slippage_bps=(0.0, 10.0),
        execution_delay_days=(1,),
        top_n=(1,),
        max_single_position_pct=(0.5,),
    )

    report = SensitivityAnalyzer().run(
        grid.scenarios(),
        lambda scenario: runner.run(scenario).metrics,
    )

    assert report.summary.scenario_count == 4
    assert report.summary.return_degradation >= 0
    assert report.summary.pass_ratio == 1.0


def test_factor_rank_runner_uses_saved_file_factor(tmp_path) -> None:
    factor_file = tmp_path / "factor.py"
    factor_file.write_text(
        """
from typing import Any

import pandas as pd


def compute(bars: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    return bars.groupby("symbol")["close"].pct_change(3)
""",
        encoding="utf-8",
    )
    registry_root = tmp_path / "factors"
    FactorRegistry(registry_root).save_factor(
        factor_id="agent_momentum_3d",
        name="Agent momentum 3d",
        version="0.1.0",
        implementation_ref=f"file:{factor_file}",
        required_columns=("symbol", "trade_date", "close"),
        lookback=3,
        created_by="agent",
    )
    runner = FactorRankResearchRunner(
        _bars(),
        FactorRankResearchConfig(
            factor_name="agent_momentum_3d",
            expected_trade_dates=tuple(sorted(_bars()["trade_date"].unique())),
            factor_registry_root=registry_root,
            top_n=1,
            max_single_position_pct=0.5,
            initial_cash=100000,
        ),
    )

    result = runner.run(SensitivityGrid(top_n=(1,)).scenarios()[0])

    assert result.trades


def test_factor_rank_runner_preindexes_bars_by_date_and_symbol() -> None:
    runner = FactorRankResearchRunner(
        _bars(),
        FactorRankResearchConfig(
            factor_name="momentum_20d",
            expected_trade_dates=tuple(sorted(_bars()["trade_date"].unique())),
            top_n=1,
            max_single_position_pct=0.5,
            initial_cash=100000,
        ),
    )

    first_date = runner.bars["trade_date"].min()
    day_bars = runner._bars_on(first_date)
    row = runner._bar_for_symbol(day_bars, "000001.SZ")

    assert day_bars.index.name == "symbol"
    assert row is not None
    assert row["symbol"] == "000001.SZ"
