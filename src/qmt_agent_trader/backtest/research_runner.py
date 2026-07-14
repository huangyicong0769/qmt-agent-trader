"""Deterministic daily-ledger research runner for factor-rank strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from qmt_agent_trader.backtest.commission import CostConfig, calculate_cost_breakdown
from qmt_agent_trader.backtest.errors import (
    BacktestAccountingError,
    BacktestDataIntegrityError,
    BacktestUniverseIntegrityError,
)
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
from qmt_agent_trader.data.integrity import require_unique_symbol_dates
from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.factors.service import compute_factor_frame
from qmt_agent_trader.universe.timeline import RollingUniverseTimeline

_CASH_EPSILON = 1e-8
_REQUIRED_CANONICAL_BAR_COLUMNS = {
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "suspended",
    "st",
    "limit_up_at_open",
    "limit_down_at_open",
}
_EXECUTION_STATE_COLUMNS = {
    "suspended",
    "st",
    "limit_up_at_open",
    "limit_down_at_open",
}


@dataclass(frozen=True)
class FactorRankResearchConfig:
    factor_name: str
    expected_trade_dates: tuple[date, ...]
    factor_registry_root: Path | None = None
    factor_registry: FactorRegistry | None = None
    top_n: int = 20
    max_single_position_pct: float = 0.10
    initial_cash: float = 1_000_000.0
    rebalance_every_n_days: int = 1
    rebalance_frequency: RebalanceFrequency = "daily"
    min_turnover_threshold: float = 0.0
    rank_buffer: int = 0
    cash_buffer_pct: float = 0.02
    lower_is_better: bool = False
    symbols_by_date: dict[str, list[str]] | None = None
    insufficient_history_by_symbol: dict[str, dict[str, int]] = field(
        default_factory=dict
    )
    base_cost_config: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if not 0 < self.max_single_position_pct <= 1:
            raise ValueError("max_single_position_pct must be in (0, 1]")
        if not 0 <= self.cash_buffer_pct < 1:
            raise ValueError("cash_buffer_pct must be in [0, 1)")
        if not 0 <= self.min_turnover_threshold <= 1:
            raise ValueError("min_turnover_threshold must be in [0, 1]")
        if self.rank_buffer < 0:
            raise ValueError("rank_buffer must be non-negative")
        if self.rebalance_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError(
                f"unsupported rebalance_frequency: {self.rebalance_frequency}"
            )
        if not self.expected_trade_dates:
            raise ValueError("expected_trade_dates cannot be empty")
        if self.expected_trade_dates != tuple(sorted(set(self.expected_trade_dates))):
            raise ValueError("expected_trade_dates must be sorted and unique")


class FactorRankResearchRunner:
    """Run after-close signals at a delayed open and value every trading day."""

    def __init__(self, bars: pd.DataFrame, config: FactorRankResearchConfig) -> None:
        self.bars = _prepare_bars(bars)
        self.config = config
        observed_dates = set(self.bars["trade_date"])
        expected_dates = set(config.expected_trade_dates)
        missing_dates = sorted(expected_dates - observed_dates)
        if missing_dates:
            raise BacktestDataIntegrityError(
                code="MISSING_EXPECTED_TRADING_SESSION",
                message="one or more expected open sessions have no market bars",
                field="trade_date",
                details={
                    "missing_dates": [f"{item:%Y-%m-%d}" for item in missing_dates]
                },
            )
        first_expected = config.expected_trade_dates[0]
        last_expected = config.expected_trade_dates[-1]
        unexpected_dates = sorted(day for day in observed_dates if day > last_expected)
        interior_non_session_dates = sorted(
            day
            for day in observed_dates
            if first_expected <= day <= last_expected and day not in expected_dates
        )
        if unexpected_dates or interior_non_session_dates:
            raise BacktestDataIntegrityError(
                code="UNEXPECTED_MARKET_SESSION",
                message=(
                    "market bars contain non-calendar dates inside or after the backtest window"
                ),
                field="trade_date",
                details={
                    "unexpected_dates": [
                        f"{item:%Y-%m-%d}"
                        for item in [*interior_non_session_dates, *unexpected_dates]
                    ]
                },
            )
        registry = config.factor_registry or (
            FactorRegistry(config.factor_registry_root)
            if config.factor_registry_root is not None
            else None
        )
        self.factor_frame = compute_factor_frame(self.bars, config.factor_name, registry=registry)
        require_unique_symbol_dates(
            self.factor_frame,
            symbol_column="symbol",
            date_column="trade_date",
            code="DUPLICATE_FACTOR_SYMBOL_DATE",
            field="factor_frame",
        )
        if config.lower_is_better:
            self.factor_frame = self.factor_frame.copy()
            self.factor_frame["factor_value"] = -pd.to_numeric(
                self.factor_frame["factor_value"],
                errors="coerce",
            )
        execution_bars = self.bars[
            self.bars["trade_date"].isin(config.expected_trade_dates)
        ].copy()
        self._bars_by_date_symbol = {
            trade_date: frame.set_index("symbol", drop=False)
            for trade_date, frame in execution_bars.groupby("trade_date", sort=True)
        }
        self._universe_timeline = (
            RollingUniverseTimeline.from_mapping(config.symbols_by_date)
            if config.symbols_by_date
            else None
        )

    def run(self, scenario: SensitivityScenario) -> FactorRankResearchResult:
        scenario.validate_for_factor_rank()
        top_n = scenario.top_n or self.config.top_n
        max_position = scenario.max_single_position_pct or self.config.max_single_position_pct
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if not 0 < max_position <= 1:
            raise ValueError("max_single_position_pct must be in (0, 1]")

        dates = self.config.expected_trade_dates
        signal_dates = select_signal_dates(dates, self.config.rebalance_frequency)
        if self.config.rebalance_every_n_days > 1:
            signal_dates = signal_dates[:: self.config.rebalance_every_n_days]
        execution_schedule = build_execution_schedule(
            dates,
            signal_dates=signal_dates,
            delay_days=scenario.execution_delay_days,
        )
        if signal_dates and not execution_schedule:
            raise BacktestDataIntegrityError(
                code="NO_EXECUTION_SESSION_AFTER_SIGNAL",
                message="no execution session exists after any signal",
                field="execution_schedule",
                details={"execution_delay_days": scenario.execution_delay_days},
            )
        scheduled_signal_dates = tuple(dict.fromkeys(execution_schedule.values()))
        available_signals, unavailable_signals = self._prepare_scheduled_signal_frames(
            scheduled_signal_dates
        )
        if scheduled_signal_dates and not available_signals:
            raise BacktestDataIntegrityError(
                code="NO_EXECUTABLE_FACTOR_SIGNALS",
                message="no scheduled factor signal is executable",
                field="factor_frame",
                details={
                    "unavailable_signals": {
                        item.isoformat(): reason
                        for item, reason in unavailable_signals.items()
                    }
                },
            )

        cash = self.config.initial_cash
        positions: dict[str, int] = {}
        trades: list[ResearchTrade] = []
        equity_points: list[ResearchEquityPoint] = []
        rebalance_points: list[ResearchRebalancePoint] = []
        rejected_orders = 0
        total_explicit_cost = 0.0
        total_slippage_cost = 0.0
        previous_selected: set[str] | None = None
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
                factors = available_signals.get(signal_date)
                if factors is None:
                    rebalance_points.append(
                        ResearchRebalancePoint(
                            signal_date=f"{signal_date:%Y-%m-%d}",
                            trade_date=f"{trade_date:%Y-%m-%d}",
                            equity_before=equity_before,
                            gross_traded_notional=0.0,
                            one_way_turnover=0.0,
                            selected_count=0,
                            retained_count=len(positions),
                            entered_count=0,
                            exited_count=0,
                            skipped=True,
                            skip_reason=unavailable_signals[signal_date],
                        )
                    )
                if factors is not None and not factors.empty:
                    selected_symbols = self._select_symbols(
                        factors=factors,
                        positions=positions,
                        top_n=top_n,
                    )
                    targets = self._target_quantities(
                        selected_symbols=selected_symbols,
                        day_bars=day_bars,
                        trade_date=trade_date,
                        equity=equity_before,
                        top_n=top_n,
                        max_position=max_position,
                        scenario=scenario,
                    )
                    before_symbols = set(positions)
                    planned_turnover = self._planned_one_way_turnover(
                        positions=positions,
                        targets=targets,
                        trade_date=trade_date,
                        day_bars=day_bars,
                        equity=equity_before,
                    )
                    skip_for_turnover = (
                        planned_turnover < self.config.min_turnover_threshold
                    )
                    if skip_for_turnover:
                        rebalance_points.append(
                            ResearchRebalancePoint(
                                signal_date=f"{signal_date:%Y-%m-%d}",
                                trade_date=f"{trade_date:%Y-%m-%d}",
                                equity_before=equity_before,
                                gross_traded_notional=0.0,
                                one_way_turnover=0.0,
                                selected_count=len(selected_symbols),
                                retained_count=len(before_symbols & set(selected_symbols)),
                                entered_count=0,
                                exited_count=0,
                                skipped=True,
                                skip_reason="below_min_turnover_threshold",
                            )
                        )
                    orders = (
                        []
                        if skip_for_turnover
                        else [
                            (symbol, targets.get(symbol, 0) - positions.get(symbol, 0))
                            for symbol in sorted(set(positions) | set(targets))
                            if targets.get(symbol, 0) != positions.get(symbol, 0)
                        ]
                    )
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
                            affordable = _max_affordable_buy_quantity(
                                cash=cash,
                                price=price,
                                desired_quantity=quantity,
                                cost_config=scaled_costs,
                            )
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
                        _assert_ledger_invariants(
                            cash=cash,
                            positions=positions,
                            trade_date=trade_date,
                        )
                        gross_notional += notional
                        total_explicit_cost += breakdown.total
                        total_slippage_cost += abs(price - reference_price) * quantity
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
                    if not skip_for_turnover:
                        after_symbols = set(positions)
                        current_selected = set(selected_symbols)
                        selection_jaccard = (
                            len(previous_selected & current_selected)
                            / max(1, len(previous_selected | current_selected))
                            if previous_selected is not None
                            else None
                        )
                        previous_selected = current_selected
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
                                selection_jaccard=selection_jaccard,
                            )
                        )

            _assert_ledger_invariants(
                cash=cash,
                positions=positions,
                trade_date=trade_date,
            )
            market_value = self._position_value_strict(
                positions,
                trade_date=trade_date,
                day_bars=day_bars,
                field="close",
                error_code="MISSING_HELD_POSITION_BAR",
            )
            equity_after = cash + market_value
            _assert_equity_invariant(
                equity=equity_after,
                trade_date=trade_date,
            )
            equity_points.append(
                ResearchEquityPoint(
                    trade_date=f"{trade_date:%Y-%m-%d}",
                    cash=cash,
                    market_value=market_value,
                    equity=equity_after,
                    stale_position_count=0,
                )
            )

        equity_curve = [point.equity for point in equity_points]
        turnover_series = [point.one_way_turnover for point in rebalance_points]
        metrics = _metrics_from_equity(
                equity_curve,
                turnover_series=turnover_series,
                diagnostic_pass=rejected_orders == 0,
            )
        final_net_equity = equity_points[-1].equity if equity_points else self.config.initial_cash
        same_trade_gross_return = (
            (final_net_equity + total_explicit_cost + total_slippage_cost)
            / self.config.initial_cash
            - 1.0
        )
        overlaps = [
            point.selection_jaccard
            for point in rebalance_points
            if point.selection_jaccard is not None and not point.skipped
        ]
        return FactorRankResearchResult(
            metrics=metrics,
            trades=tuple(trades),
            equity_points=tuple(equity_points),
            rebalance_points=tuple(rebalance_points),
            data_quality=ResearchDataQuality(
                validated_valuation_dates=len(equity_points),
                rejected_order_count=rejected_orders,
                scheduled_rebalance_count=len(execution_schedule),
                available_signal_count=len(available_signals),
                signal_unavailable_count=len(unavailable_signals),
                insufficient_history_by_symbol=dict(
                    self.config.insufficient_history_by_symbol
                ),
            ),
            rejected_orders=rejected_orders,
            total_explicit_cost=total_explicit_cost,
            total_slippage_cost=total_slippage_cost,
            same_trade_gross_return=same_trade_gross_return,
            average_top_n_overlap=(sum(overlaps) / len(overlaps) if overlaps else None),
        )

    def _bars_on(self, trade_date: object) -> pd.DataFrame:
        return self._bars_by_date_symbol.get(trade_date, pd.DataFrame(columns=self.bars.columns))

    def _prepare_scheduled_signal_frames(
        self,
        signal_dates: tuple[date, ...],
    ) -> tuple[dict[date, pd.DataFrame], dict[date, str]]:
        raw_by_date = {
            trade_date: frame.copy()
            for trade_date, frame in self.factor_frame.groupby("trade_date")
        }
        available: dict[date, pd.DataFrame] = {}
        unavailable: dict[date, str] = {}
        for signal_date in signal_dates:
            raw = raw_by_date.get(signal_date)
            if raw is None:
                unavailable[signal_date] = "factor_signal_date_missing"
                continue
            clean = raw.dropna(subset=["factor_value"]).sort_values(
                "factor_value",
                ascending=False,
            )
            if clean.empty:
                unavailable[signal_date] = "factor_signal_all_null"
                continue
            clean = self._filter_factors_for_history(clean, signal_date)
            if clean.empty:
                unavailable[signal_date] = (
                    "factor_signal_empty_after_history_filter"
                )
                continue
            filtered = self._filter_factors_for_universe(clean, signal_date)
            if filtered is None or filtered.empty:
                unavailable[signal_date] = "factor_signal_empty_after_universe_filter"
                continue
            available[signal_date] = filtered
        return available, unavailable

    def _filter_factors_for_history(
        self,
        factors: pd.DataFrame,
        signal_date: date,
    ) -> pd.DataFrame:
        insufficient = self.config.insufficient_history_by_symbol
        if not insufficient:
            return factors
        history = self.bars[self.bars["trade_date"] <= signal_date]
        counts = history.groupby("symbol")["trade_date"].nunique().to_dict()
        ready = [
            symbol
            for symbol in factors["symbol"].astype(str)
            if symbol not in insufficient
            or int(counts.get(symbol, 0))
            >= int(insufficient[symbol]["required_sessions"])
        ]
        return factors[factors["symbol"].astype(str).isin(ready)].copy()

    def _filter_factors_for_universe(
        self,
        factors: pd.DataFrame | None,
        signal_date: date,
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
        if self._universe_timeline is None:
            return filtered
        symbols = self._universe_timeline.membership_as_of(signal_date)
        if not symbols:
            raise BacktestUniverseIntegrityError(
                code="ROLLING_UNIVERSE_EMPTY_AS_OF_SIGNAL",
                message="resolved rolling-universe membership is empty",
                trade_date=f"{signal_date:%Y-%m-%d}",
                field="symbols_by_date",
            )
        return filtered[filtered["symbol"].astype(str).isin(symbols)].copy()

    @staticmethod
    def _bar_for_symbol(day_bars: pd.DataFrame, symbol: str) -> pd.Series | None:
        if day_bars.empty or symbol not in day_bars.index:
            return None
        match = day_bars.loc[symbol]
        if isinstance(match, pd.DataFrame):
            raise RuntimeError("duplicate symbol bar reached lookup after uniqueness validation")
        return match if isinstance(match, pd.Series) else None

    @staticmethod
    def _can_execute(bar: pd.Series, side: Side) -> bool:
        if bool(bar["suspended"]):
            return False
        if side == Side.BUY:
            return not bool(bar["st"]) and not bool(bar["limit_up_at_open"])
        return not bool(bar["limit_down_at_open"])

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
        selected_symbols: list[str],
        day_bars: pd.DataFrame,
        trade_date: date,
        equity: float,
        top_n: int,
        max_position: float,
        scenario: SensitivityScenario,
    ) -> dict[str, int]:
        if not selected_symbols:
            return {}
        investable_weight = max(0.0, 1.0 - self.config.cash_buffer_pct)
        weight = min(investable_weight / len(selected_symbols), max_position)
        targets: dict[str, int] = {}
        for symbol in selected_symbols:
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

    def _select_symbols(
        self,
        *,
        factors: pd.DataFrame,
        positions: dict[str, int],
        top_n: int,
    ) -> list[str]:
        ranked_symbols = factors["symbol"].astype(str).tolist()
        retention_set = set(ranked_symbols[: top_n + self.config.rank_buffer])
        retained = [symbol for symbol in positions if symbol in retention_set]
        vacancies = max(0, top_n - len(retained))
        new_entries = [symbol for symbol in ranked_symbols if symbol not in retained][:vacancies]
        return retained + new_entries

    def _planned_one_way_turnover(
        self,
        *,
        positions: dict[str, int],
        targets: dict[str, int],
        trade_date: date,
        day_bars: pd.DataFrame,
        equity: float,
    ) -> float:
        if equity <= 0:
            return 0.0
        symbols = set(positions) | set(targets)
        current_weights: dict[str, float] = {}
        target_weights: dict[str, float] = {}
        for symbol in symbols:
            price = self._required_price(
                symbol=symbol,
                trade_date=trade_date,
                day_bars=day_bars,
                field="open",
                error_code="MISSING_EXECUTION_BAR",
            )
            current_weights[symbol] = positions.get(symbol, 0) * price / equity
            target_weights[symbol] = targets.get(symbol, 0) * price / equity
        return 0.5 * sum(
            abs(target_weights[symbol] - current_weights[symbol]) for symbol in symbols
        )

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
    missing_state = sorted(_EXECUTION_STATE_COLUMNS.difference(bars.columns))
    if missing_state:
        raise BacktestDataIntegrityError(
            code="MISSING_EXECUTION_STATE_COLUMNS",
            message="canonical bars are missing opening execution-state columns",
            field="bars",
            details={"missing_columns": missing_state},
        )
    missing = sorted(_REQUIRED_CANONICAL_BAR_COLUMNS.difference(bars.columns))
    if missing:
        raise BacktestDataIntegrityError(
            code="MISSING_CANONICAL_BAR_COLUMNS",
            message="canonical bars are missing required columns",
            field="bars",
            details={"missing_columns": missing},
        )
    data = bars.copy()
    data["symbol"] = data["symbol"].astype(str)
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
    require_unique_symbol_dates(
        data,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_SYMBOL_DATE_BAR",
        field="bars",
    )
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in sorted(_EXECUTION_STATE_COLUMNS):
        if data[column].isna().any():
            raise BacktestDataIntegrityError(
                code="UNKNOWN_EXECUTION_STATE",
                message="canonical execution state contains unknown values",
                field=column,
            )
        data[column] = data[column].astype(bool)
    return data.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _max_affordable_buy_quantity(
    *,
    cash: float,
    price: float,
    desired_quantity: int,
    cost_config: CostConfig,
) -> int:
    desired_lots = max(0, desired_quantity // 100)
    low = 0
    high = desired_lots
    while low < high:
        middle = (low + high + 1) // 2
        quantity = middle * 100
        notional = quantity * price
        total = notional + calculate_cost_breakdown(
            notional,
            Side.BUY,
            cost_config,
        ).total
        if total <= cash + _CASH_EPSILON:
            low = middle
        else:
            high = middle - 1
    return low * 100


def _assert_ledger_invariants(
    *,
    cash: float,
    positions: dict[str, int],
    trade_date: date,
) -> None:
    if not math.isfinite(cash):
        raise BacktestAccountingError(
            code="NON_FINITE_CASH",
            message="cash must remain finite",
            trade_date=f"{trade_date:%Y-%m-%d}",
            field="cash",
            details={"cash": cash},
        )
    if cash < -_CASH_EPSILON:
        raise BacktestAccountingError(
            code="NEGATIVE_CASH_AFTER_TRADE",
            message="post-trade cash violated the non-negative invariant",
            trade_date=f"{trade_date:%Y-%m-%d}",
            field="cash",
            details={"cash": cash, "tolerance": _CASH_EPSILON},
        )
    invalid_positions = {
        symbol: quantity for symbol, quantity in positions.items() if quantity <= 0
    }
    if invalid_positions:
        raise BacktestAccountingError(
            code="INVALID_POSITION_QUANTITY",
            message="completed ledger positions must have positive quantities",
            trade_date=f"{trade_date:%Y-%m-%d}",
            field="positions",
            symbols=tuple(sorted(invalid_positions)),
            details={"positions": invalid_positions},
        )


def _assert_equity_invariant(*, equity: float, trade_date: date) -> None:
    if not math.isfinite(equity) or equity < -_CASH_EPSILON:
        raise BacktestAccountingError(
            code="INVALID_EQUITY_VALUE",
            message="daily equity must be finite and non-negative",
            trade_date=f"{trade_date:%Y-%m-%d}",
            field="equity",
            details={"equity": equity},
        )


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
