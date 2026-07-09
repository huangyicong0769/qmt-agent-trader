"""Executable research runners for candidate strategy evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from qmt_agent_trader.backtest.commission import CostConfig, calculate_cost
from qmt_agent_trader.backtest.sensitivity import SensitivityMetrics, SensitivityScenario
from qmt_agent_trader.backtest.slippage import fixed_bps_slippage
from qmt_agent_trader.core.types import Side
from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.factors.service import compute_factor_frame


@dataclass(frozen=True)
class FactorRankResearchConfig:
    factor_name: str
    factor_registry_root: Path | None = None
    factor_registry: FactorRegistry | None = None
    top_n: int = 20
    max_single_position_pct: float = 0.10
    initial_cash: float = 1_000_000.0
    rebalance_every_n_days: int = 1
    symbols_by_date: dict[str, list[str]] | None = None
    base_cost_config: CostConfig = field(default_factory=CostConfig)


@dataclass(frozen=True)
class ResearchTrade:
    signal_date: str
    trade_date: str
    symbol: str
    side: Side
    quantity: int
    price: float
    notional: float
    cost: float


@dataclass(frozen=True)
class FactorRankResearchResult:
    metrics: SensitivityMetrics
    trades: tuple[ResearchTrade, ...]
    equity_curve: tuple[float, ...]
    turnover_series: tuple[float, ...]
    rejected_orders: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.as_dict(),
            "trades": [trade.__dict__ for trade in self.trades],
            "equity_curve": list(self.equity_curve),
            "turnover_series": list(self.turnover_series),
            "rejected_orders": self.rejected_orders,
        }


class FactorRankResearchRunner:
    """Run a deterministic T+delay daily factor-rank portfolio simulation."""

    def __init__(self, bars: pd.DataFrame, config: FactorRankResearchConfig) -> None:
        self.bars = _prepare_bars(bars)
        self.config = config
        registry = config.factor_registry or (
            FactorRegistry(config.factor_registry_root)
            if config.factor_registry_root is not None
            else None
        )
        self.factor_frame = compute_factor_frame(self.bars, config.factor_name, registry=registry)
        self._bars_by_date_symbol = {
            trade_date: frame.set_index("symbol", drop=False)
            for trade_date, frame in self.bars.groupby("trade_date", sort=True)
        }

    def run(self, scenario: SensitivityScenario) -> FactorRankResearchResult:
        top_n = scenario.top_n or self.config.top_n
        max_position = scenario.max_single_position_pct or self.config.max_single_position_pct
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if max_position <= 0 or max_position > 1:
            raise ValueError("max_single_position_pct must be in (0, 1]")

        cash = self.config.initial_cash
        positions: dict[str, int] = {}
        trades: list[ResearchTrade] = []
        equity_curve = [cash]
        turnover_series: list[float] = []
        rejected_orders = 0
        dates = sorted(self.bars["trade_date"].unique())
        factor_by_date = {
            trade_date: frame.dropna(subset=["factor_value"]).sort_values(
                "factor_value", ascending=False
            )
            for trade_date, frame in self.factor_frame.groupby("trade_date")
        }

        for index, signal_date in enumerate(dates):
            if index % self.config.rebalance_every_n_days != 0:
                continue
            execution_index = index + scenario.execution_delay_days
            if execution_index >= len(dates):
                continue
            execution_date = dates[execution_index]
            factors = factor_by_date.get(signal_date)
            factors = self._filter_factors_for_universe(factors, signal_date)
            if factors is None or factors.empty:
                continue
            day_bars = self._bars_on(execution_date)
            if day_bars.empty:
                continue

            equity_before = cash + self._mark_to_market(positions, day_bars)
            targets = self._target_quantities(
                factors=factors,
                day_bars=day_bars,
                equity=equity_before,
                top_n=top_n,
                max_position=max_position,
                scenario=scenario,
            )
            symbols = sorted(set(positions) | set(targets))
            orders = [
                (symbol, targets.get(symbol, 0) - positions.get(symbol, 0))
                for symbol in symbols
                if targets.get(symbol, 0) - positions.get(symbol, 0) != 0
            ]
            orders.sort(key=lambda item: 0 if item[1] < 0 else 1)
            traded_notional = 0.0
            for symbol, delta in orders:
                bar = self._bar_for_symbol(day_bars, symbol)
                if bar is None:
                    rejected_orders += 1
                    continue
                side = Side.BUY if delta > 0 else Side.SELL
                quantity = abs(delta)
                price = fixed_bps_slippage(
                    float(bar["open"]),
                    side,
                    bps=scenario.slippage_bps,
                )
                notional = quantity * price
                cost = calculate_cost(
                    notional,
                    side,
                    _scaled_cost_config(self.config.base_cost_config, scenario.cost_multiplier),
                )
                if side == Side.BUY and notional + cost > cash:
                    affordable = int(cash / max(price, 1e-9) // 100 * 100)
                    quantity = min(quantity, affordable)
                    if quantity <= 0:
                        rejected_orders += 1
                        continue
                    notional = quantity * price
                    cost = calculate_cost(
                        notional,
                        side,
                        _scaled_cost_config(
                            self.config.base_cost_config, scenario.cost_multiplier
                        ),
                    )
                cash, positions = self._apply_trade(
                    cash,
                    positions,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    notional=notional,
                    cost=cost,
                )
                traded_notional += notional
                trades.append(
                    ResearchTrade(
                        signal_date=f"{signal_date:%Y-%m-%d}",
                        trade_date=f"{execution_date:%Y-%m-%d}",
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        notional=notional,
                        cost=cost,
                    )
                )
            equity_after = cash + self._mark_to_market(positions, day_bars)
            equity_curve.append(equity_after)
            turnover_series.append(traded_notional / equity_before if equity_before > 0 else 0.0)

        metrics = _metrics_from_equity(
            equity_curve,
            turnover_series=turnover_series,
            diagnostic_pass=rejected_orders == 0,
        )
        return FactorRankResearchResult(
            metrics=metrics,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
            turnover_series=tuple(turnover_series),
            rejected_orders=rejected_orders,
        )

    def _bars_on(self, trade_date: object) -> pd.DataFrame:
        return self._bars_by_date_symbol.get(trade_date, pd.DataFrame(columns=self.bars.columns))

    def _filter_factors_for_universe(
        self,
        factors: pd.DataFrame | None,
        signal_date: object,
    ) -> pd.DataFrame | None:
        if factors is None or factors.empty or not self.config.symbols_by_date:
            return factors
        key = f"{signal_date:%Y%m%d}" if hasattr(signal_date, "strftime") else str(signal_date)
        symbols = self.config.symbols_by_date.get(key)
        if symbols is None:
            return factors.iloc[0:0].copy()
        return factors[factors["symbol"].astype(str).isin(symbols)].copy()

    @staticmethod
    def _bar_for_symbol(day_bars: pd.DataFrame, symbol: str) -> pd.Series | None:
        if day_bars.empty:
            return None
        if day_bars.index.name == "symbol":
            if symbol not in day_bars.index:
                return None
            match = day_bars.loc[symbol]
            row = match.iloc[0] if isinstance(match, pd.DataFrame) else match
            return row if isinstance(row, pd.Series) else None
        matches = day_bars[day_bars["symbol"] == symbol]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return row if isinstance(row, pd.Series) else None

    def _target_quantities(
        self,
        *,
        factors: pd.DataFrame,
        day_bars: pd.DataFrame,
        equity: float,
        top_n: int,
        max_position: float,
        scenario: SensitivityScenario,
    ) -> dict[str, int]:
        selected_bars: list[tuple[str, pd.Series]] = []
        for symbol in factors["symbol"].astype(str).tolist():
            bar = self._bar_for_symbol(day_bars, symbol)
            if bar is not None:
                selected_bars.append((symbol, bar))
            if len(selected_bars) >= top_n:
                break
        if not selected_bars:
            return {}
        weight = min(1.0 / len(selected_bars), max_position)
        targets: dict[str, int] = {}
        for symbol, bar in selected_bars:
            price = fixed_bps_slippage(
                float(bar["open"]),
                Side.BUY,
                bps=scenario.slippage_bps,
            )
            targets[symbol] = int((equity * weight) / price // 100 * 100)
        return targets

    @staticmethod
    def _mark_to_market(positions: dict[str, int], day_bars: pd.DataFrame) -> float:
        market_value = 0.0
        for symbol, quantity in positions.items():
            if day_bars.empty:
                continue
            if day_bars.index.name == "symbol":
                if symbol not in day_bars.index:
                    continue
                match = day_bars.loc[symbol]
                row = match.iloc[0] if isinstance(match, pd.DataFrame) else match
            else:
                matches = day_bars[day_bars["symbol"] == symbol]
                if matches.empty:
                    continue
                row = matches.iloc[0]
            market_value += quantity * float(row.close)
        return market_value

    @staticmethod
    def _apply_trade(
        cash: float,
        positions: dict[str, int],
        *,
        symbol: str,
        side: Side,
        quantity: int,
        notional: float,
        cost: float,
    ) -> tuple[float, dict[str, int]]:
        updated = dict(positions)
        if side == Side.BUY:
            cash -= notional + cost
            updated[symbol] = updated.get(symbol, 0) + quantity
        else:
            cash += notional - cost
            updated[symbol] = updated.get(symbol, 0) - quantity
            if updated[symbol] <= 0:
                updated.pop(symbol)
        return cash, updated


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "trade_date", "open", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    data = bars.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
    for column in ["high", "low", "volume", "amount", "turnover"]:
        if column not in data.columns:
            data[column] = 0.0
    for column in ["suspended", "limit_up", "limit_down", "st"]:
        if column not in data.columns:
            data[column] = False
    data = data[~data["suspended"] & ~data["st"]]
    return data.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _scaled_cost_config(config: CostConfig, multiplier: float) -> CostConfig:
    return CostConfig(
        commission_rate=config.commission_rate * multiplier,
        stamp_tax_rate=config.stamp_tax_rate * multiplier,
        transfer_fee_rate=config.transfer_fee_rate * multiplier,
        min_commission=config.min_commission * multiplier,
    )


def _metrics_from_equity(
    equity_curve: list[float],
    *,
    turnover_series: list[float],
    diagnostic_pass: bool,
) -> SensitivityMetrics:
    if len(equity_curve) < 2:
        return SensitivityMetrics(total_return=0.0, turnover=0.0, diagnostic_pass=False)
    total_return = equity_curve[-1] / equity_curve[0] - 1.0
    returns = [
        equity_curve[index] / equity_curve[index - 1] - 1.0
        for index in range(1, len(equity_curve))
        if equity_curve[index - 1] > 0
    ]
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = min(max_drawdown, value / peak - 1.0)
    return SensitivityMetrics(
        total_return=total_return,
        sharpe=_simple_sharpe(returns),
        max_drawdown=max_drawdown,
        turnover=sum(turnover_series) / len(turnover_series) if turnover_series else 0.0,
        diagnostic_pass=diagnostic_pass,
    )


def _simple_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    series = pd.Series(returns)
    volatility = float(series.std(ddof=0))
    if volatility == 0:
        return 0.0
    return float(series.mean() / volatility)
