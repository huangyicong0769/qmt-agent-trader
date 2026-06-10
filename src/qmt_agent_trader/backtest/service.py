"""Backtest service backed by data-lake bars."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qmt_agent_trader.backtest.engine import DailyBacktestEngine
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.core.types import Side
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake


@dataclass(frozen=True)
class BacktestRunSummary:
    run_id: str
    symbol: str
    signal_date: str
    quantity: int
    fills: int
    execution_dates: list[str]
    leakage_valid: bool
    report_path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "completed",
            "run_id": self.run_id,
            "symbol": self.symbol,
            "signal_date": self.signal_date,
            "quantity": self.quantity,
            "fills": self.fills,
            "execution_dates": self.execution_dates,
            "leakage_valid": self.leakage_valid,
            "report_path": self.report_path,
        }


def run_single_symbol_backtest(
    lake: DataLake,
    *,
    symbol: str | None = None,
    signal_date: str | None = None,
    quantity: int = 100,
) -> BacktestRunSummary:
    bars = load_daily_bars(lake)
    if bars.empty:
        raise ValueError("no daily bars found in data lake; run data update first")

    chosen_symbol = symbol or str(bars["symbol"].iloc[0])
    symbol_bars = bars[bars["symbol"] == chosen_symbol].sort_values("trade_date")
    if len(symbol_bars) < 2:
        raise ValueError(f"not enough bars for T+1 backtest: {chosen_symbol}")

    chosen_signal_date = _choose_signal_date(symbol_bars, signal_date)
    result = DailyBacktestEngine().run_one_signal(
        bars,
        symbol=chosen_symbol,
        signal_date=chosen_signal_date,
        side=Side.BUY,
        quantity=quantity,
    )
    return BacktestRunSummary(
        run_id=new_id("bt"),
        symbol=chosen_symbol,
        signal_date=chosen_signal_date,
        quantity=quantity,
        fills=len(result.fills),
        execution_dates=[f"{fill.trade_date:%Y-%m-%d}" for fill in result.fills],
        leakage_valid=result.leakage_valid,
    )


def run_backtest_report(
    lake: DataLake,
    *,
    reports_dir: Path,
    symbol: str | None = None,
    signal_date: str | None = None,
    quantity: int = 100,
    config_path: str = "configs/backtest.yaml",
) -> BacktestRunSummary:
    summary = run_single_symbol_backtest(
        lake,
        symbol=symbol,
        signal_date=signal_date,
        quantity=quantity,
    )
    report = _build_report(summary, config_path=config_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{summary.run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return BacktestRunSummary(
        run_id=summary.run_id,
        symbol=summary.symbol,
        signal_date=summary.signal_date,
        quantity=summary.quantity,
        fills=summary.fills,
        execution_dates=summary.execution_dates,
        leakage_valid=summary.leakage_valid,
        report_path=str(report_path),
    )


def compare_backtest_reports(reports_dir: Path, *, limit: int = 10) -> dict[str, object]:
    if not reports_dir.exists():
        return {"status": "empty", "runs": []}
    paths = sorted(
        reports_dir.glob("bt_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths[:limit]]
    return {"status": "compared", "runs": runs}


def _choose_signal_date(symbol_bars: pd.DataFrame, signal_date: str | None) -> str:
    if signal_date is not None:
        return f"{pd.to_datetime(signal_date).date():%Y-%m-%d}"
    value = symbol_bars.iloc[0]["trade_date"]
    return f"{pd.to_datetime(value).date():%Y-%m-%d}"


def _build_report(summary: BacktestRunSummary, *, config_path: str) -> dict[str, object]:
    return {
        "run_id": summary.run_id,
        "created_at": shanghai_now_iso(),
        "config_snapshot": {"config_path": config_path, "mode": "daily_t_plus_1"},
        "data_version": "data_lake_current",
        "strategy_version": "single_symbol_smoke_v1",
        "factor_version": None,
        "performance_report": {"fills": summary.fills},
        "turnover_report": {"orders": summary.fills},
        "cost_report": {"mode": "simulated_cost_model"},
        "leakage_report": {"valid": summary.leakage_valid, "checks": ["signal_before_execution"]},
        "holdings_report": {"symbol": summary.symbol, "quantity": summary.quantity},
        "trade_blotter": [
            {
                "symbol": summary.symbol,
                "execution_date": execution_date,
                "quantity": summary.quantity,
            }
            for execution_date in summary.execution_dates
        ],
    }
