"""Deterministic daily-ledger research runner for factor-rank strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from qmt_agent_trader.backtest.commission import CostConfig, calculate_cost_breakdown
from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.backtest.rebalance import (
    RebalanceFrequency,
    build_execution_schedule,
    select_signal_dates,
)
from qmt_agent_trader.backtest.research_models import (
    FactorRankResearchResult,
    ResearchDataQuality,
    ResearchEquityPoint,
    ResearchRebalancePoint,
    ResearchTrade,
)
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
    rebalance_frequency: RebalanceFrequency = "daily"
    symbols_by_date: dict[str, list[str]] | None = None
    base_cost_config: CostConfig = field(default_factory=CostConfig)


class FactorRankResearchRunner:
    """Run after-close signals at a delayed open and value every trading day."""

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
        if not 0 < max_position <= 1:
            raise ValueError("max_single_position_pct must be in (0, 1]")

        dates: tuple[date, ...] = tuple(sorted(self.bars["trade_date"].unique()))
        signal_dates = select_signal_dates(dates, self.config.rebalance_frequency)
        if self.config.rebalance_every_n_days > 1:
            signal_dates = signal_dates[:: self.config.rebalance_every_n_days]
        execution_schedule = build_execution_schedule(
            dates,
            signal_dates=signal_dates,
            delay_days=scenario.execution_delay_days,
        )
        factor_by_date = {
            trade_date: frame.dropna(subset=["factor_value"]).sort_values(
                "factor_value", ascending=False
            )
            for trade_date, frame in self.factor_frame.groupby("trade_date")
        }

        cash = self.config.initial_cash
        positions: dict[str, int] = {}
        trades: list[ResearchTrade] = []
        equity_points: list[ResearchEquityPoint] = []
        rebalance_points: list[ResearchRebalancePoint] = []
        rejected_orders = 0
        scaled_costs = _scaled_cost_config(
            self.config.base_cost_config,
            scenario.cost_multiplier,
        )

        for trade_date in dates:
            day_bars = self._bars_on(trade_date)
            equity_before = cash + self._position_value_strict(
                positions,
                trade_date=trade_date,
                day_bars=day_bars,
                field="open",
                error_code="MISSING_HELD_POSITION_BAR",
            )
            signal_date = execution_schedule.get(trade_date)
            if signal_date is not None:
                factors = self._filter_factors_for_universe(
                    factor_by_date.get(signal_date),
                    signal_date,
                )
                if factors is not None and not factors.empty:
                    targets = self._target_quantities(
                        factors=factors,
                        day_bars=day_bars,
                        trade_date=trade_date,
                        equity=equity_before,
                        top_n=top_n,
                        max_position=max_position,
                        scenario=scenario,
                    )
                    before_symbols = set(positions)
                    orders = [
                        (symbol, targets.get(symbol, 0) - positions.get(symbol, 0))
                        for symbol in sorted(set(positions) | set(targets))
                        if targets.get(symbol, 0) != positions.get(symbol, 0)
                    ]
                    orders.sort(key=lambda item: item[1] > 0)
                    self._validate_order_prices(orders, trade_date, day_bars)
                    gross_notional = 0.0
                    for symbol, delta in orders:
                        bar = self._bar_for_symbol(day_bars, symbol)
                        assert bar is not None
                        side = Side.BUY if delta > 0 else Side.SELL
                        if not self._can_execute(bar, side):
                            rejected_orders += 1
                            continue
                        quantity = abs(delta)
                        reference_price = self._required_price(
                            symbol=symbol,
                            trade_date=trade_date,
                            day_bars=day_bars,
                            field="open",
                            error_code="MISSING_EXECUTION_BAR",
                        )
                        price = fixed_bps_slippage(
                            reference_price,
                            side,
                            bps=scenario.slippage_bps,
                        )
                        notional = quantity * price
                        breakdown = calculate_cost_breakdown(notional, side, scaled_costs)
                        if side == Side.BUY and notional + breakdown.total > cash:
                            affordable = int(cash / max(price, 1e-9) // 100 * 100)
                            quantity = min(quantity, affordable)
                            if quantity <= 0:
                                rejected_orders += 1
                                continue
                            notional = quantity * price
                            breakdown = calculate_cost_breakdown(notional, side, scaled_costs)
                        cash, positions = self._apply_trade(
                            cash,
                            positions,
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            notional=notional,
                            cost=breakdown.total,
                        )
                        gross_notional += notional
                        trades.append(
                            ResearchTrade(
                                signal_date=f"{signal_date:%Y-%m-%d}",
                                trade_date=f"{trade_date:%Y-%m-%d}",
                                symbol=symbol,
                                side=side,
                                quantity=quantity,
                                reference_price=reference_price,
                                price=price,
                                notional=notional,
                                commission=breakdown.commission,
                                stamp_tax=breakdown.stamp_tax,
                                transfer_fee=breakdown.transfer_fee,
                                slippage_cost=abs(price - reference_price) * quantity,
                                cost=breakdown.total,
                            )
                        )
                    after_symbols = set(positions)
                    rebalance_points.append(
                        ResearchRebalancePoint(
                            signal_date=f"{signal_date:%Y-%m-%d}",
                            trade_date=f"{trade_date:%Y-%m-%d}",
                            equity_before=equity_before,
                            gross_traded_notional=gross_notional,
                            one_way_turnover=(
                                gross_notional / (2.0 * equity_before)
                                if equity_before > 0
                                else 0.0
                            ),
                            selected_count=len(targets),
                            retained_count=len(before_symbols & after_symbols),
                            entered_count=len(after_symbols - before_symbols),
                            exited_count=len(before_symbols - after_symbols),
                        )
                    )

            market_value = self._position_value_strict(
                positions,
                trade_date=trade_date,
                day_bars=day_bars,
                field="close",
                error_code="MISSING_HELD_POSITION_BAR",
            )
            equity_points.append(
                ResearchEquityPoint(
                    trade_date=f"{trade_date:%Y-%m-%d}",
                    cash=cash,
                    market_value=market_value,
                    equity=cash + market_value,
                    stale_position_count=0,
                )
            )

        equity_curve = [point.equity for point in equity_points]
        turnover_series = [point.one_way_turnover for point in rebalance_points]
        return FactorRankResearchResult(
            metrics=_metrics_from_equity(
                equity_curve,
                turnover_series=turnover_series,
                diagnostic_pass=rejected_orders == 0,
            ),
            trades=tuple(trades),
            equity_points=tuple(equity_points),
            rebalance_points=tuple(rebalance_points),
            data_quality=ResearchDataQuality(
                validated_valuation_dates=len(equity_points),
                rejected_order_count=rejected_orders,
            ),
            rejected_orders=rejected_orders,
        )

    def _bars_on(self, trade_date: object) -> pd.DataFrame:
        return self._bars_by_date_symbol.get(trade_date, pd.DataFrame(columns=self.bars.columns))

    def _filter_factors_for_universe(
        self,
        factors: pd.DataFrame | None,
        signal_date: object,
    ) -> pd.DataFrame | None:
        if factors is None or factors.empty:
            return factors
        filtered = factors.copy()
        signal_bars = self._bars_on(signal_date)
        eligible = [
            symbol
            for symbol in filtered["symbol"].astype(str)
            if (bar := self._bar_for_symbol(signal_bars, symbol)) is not None
            and not bool(bar.get("suspended", False))
            and not bool(bar.get("st", False))
        ]
        filtered = filtered[filtered["symbol"].astype(str).isin(eligible)]
        if not self.config.symbols_by_date:
            return filtered
        key = f"{signal_date:%Y%m%d}" if hasattr(signal_date, "strftime") else str(signal_date)
        symbols = self.config.symbols_by_date.get(key)
        if symbols is None:
            return filtered.iloc[0:0].copy()
        return filtered[filtered["symbol"].astype(str).isin(symbols)].copy()

    @staticmethod
    def _bar_for_symbol(day_bars: pd.DataFrame, symbol: str) -> pd.Series | None:
        if day_bars.empty or symbol not in day_bars.index:
            return None
        match = day_bars.loc[symbol]
        row = match.iloc[0] if isinstance(match, pd.DataFrame) else match
        return row if isinstance(row, pd.Series) else None

    @staticmethod
    def _can_execute(bar: pd.Series, side: Side) -> bool:
        if bool(bar.get("suspended", False)):
            return False
        if side == Side.BUY:
            return not bool(bar.get("st", False)) and not bool(bar.get("limit_up", False))
        return not bool(bar.get("limit_down", False))

    def _required_price(
        self,
        *,
        symbol: str,
        trade_date: date,
        day_bars: pd.DataFrame,
        field: str,
        error_code: str,
    ) -> float:
        row = self._bar_for_symbol(day_bars, symbol)
        if row is None:
            raise BacktestDataIntegrityError(
                code=error_code,
                trade_date=f"{trade_date:%Y-%m-%d}",
                symbols=(symbol,),
                field=field,
                message="required symbol-date bar is absent",
            )
        value = pd.to_numeric(row.get(field), errors="coerce")
        if pd.isna(value) or not math.isfinite(float(value)) or float(value) <= 0:
            raise BacktestDataIntegrityError(
                code="INVALID_REQUIRED_PRICE",
                trade_date=f"{trade_date:%Y-%m-%d}",
                symbols=(symbol,),
                field=field,
                message=f"required {field} price is null, non-finite or non-positive",
            )
        return float(value)

    def _position_value_strict(
        self,
        positions: dict[str, int],
        *,
        trade_date: date,
        day_bars: pd.DataFrame,
        field: str,
        error_code: str,
    ) -> float:
        return sum(
            quantity
            * self._required_price(
                symbol=symbol,
                trade_date=trade_date,
                day_bars=day_bars,
                field=field,
                error_code=error_code,
            )
            for symbol, quantity in positions.items()
        )

    def _validate_order_prices(
        self,
        orders: list[tuple[str, int]],
        trade_date: date,
        day_bars: pd.DataFrame,
    ) -> None:
        for symbol, _delta in orders:
            self._required_price(
                symbol=symbol,
                trade_date=trade_date,
                day_bars=day_bars,
                field="open",
                error_code="MISSING_EXECUTION_BAR",
            )

    def _target_quantities(
        self,
        *,
        factors: pd.DataFrame,
        day_bars: pd.DataFrame,
        trade_date: date,
        equity: float,
        top_n: int,
        max_position: float,
        scenario: SensitivityScenario,
    ) -> dict[str, int]:
        selected = factors["symbol"].astype(str).tolist()[:top_n]
        if not selected:
            return {}
        weight = min(1.0 / len(selected), max_position)
        targets: dict[str, int] = {}
        for symbol in selected:
            reference_price = self._required_price(
                symbol=symbol,
                trade_date=trade_date,
                day_bars=day_bars,
                field="open",
                error_code="MISSING_EXECUTION_BAR",
            )
            price = fixed_bps_slippage(reference_price, Side.BUY, bps=scenario.slippage_bps)
            targets[symbol] = int((equity * weight) / price // 100 * 100)
        return targets

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
    data["symbol"] = data["symbol"].astype(str)
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
    for column in ["high", "low", "volume", "amount", "turnover"]:
        if column not in data.columns:
            data[column] = 0.0
    for column in ["suspended", "limit_up", "limit_down", "st"]:
        if column not in data.columns:
            data[column] = False
        data[column] = data[column].fillna(False).astype(bool)
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
        sharpe=_annualized_sharpe(returns),
        max_drawdown=max_drawdown,
        turnover=sum(turnover_series) / len(turnover_series) if turnover_series else 0.0,
        diagnostic_pass=diagnostic_pass,
    )


def _annualized_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    series = pd.Series(returns, dtype="float64")
    volatility = float(series.std(ddof=0))
    if volatility == 0.0:
        return 0.0
    return float(series.mean() / volatility * (252.0**0.5))
