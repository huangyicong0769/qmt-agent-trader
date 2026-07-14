"""Typed evidence emitted by deterministic research backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from qmt_agent_trader.backtest.sensitivity import SensitivityMetrics
from qmt_agent_trader.core.types import Side


@dataclass(frozen=True)
class ResearchTrade:
    signal_date: str
    trade_date: str
    symbol: str
    side: Side
    quantity: int
    reference_price: float
    price: float
    notional: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    slippage_cost: float
    cost: float
    reason: str = "rebalance"

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["side"] = self.side.value
        return payload


@dataclass(frozen=True)
class ResearchEquityPoint:
    trade_date: str
    cash: float
    market_value: float
    equity: float
    stale_position_count: int
    stale_market_value: float = 0.0


@dataclass(frozen=True)
class ResearchRebalancePoint:
    signal_date: str
    trade_date: str
    equity_before: float
    gross_traded_notional: float
    one_way_turnover: float
    selected_count: int
    retained_count: int
    entered_count: int
    exited_count: int
    skipped: bool = False
    skip_reason: str | None = None
    selection_jaccard: float | None = None


@dataclass(frozen=True)
class ResearchDataQuality:
    validated_valuation_dates: int = 0
    low_cross_section_dates: tuple[str, ...] = ()
    rejected_order_count: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorRankResearchResult:
    metrics: SensitivityMetrics
    trades: tuple[ResearchTrade, ...]
    equity_points: tuple[ResearchEquityPoint, ...]
    rebalance_points: tuple[ResearchRebalancePoint, ...]
    data_quality: ResearchDataQuality
    rejected_orders: int = 0
    total_explicit_cost: float = 0.0
    total_slippage_cost: float = 0.0
    same_trade_gross_return: float = 0.0
    average_top_n_overlap: float | None = None

    @property
    def equity_curve(self) -> tuple[float, ...]:
        return tuple(point.equity for point in self.equity_points)

    @property
    def turnover_series(self) -> tuple[float, ...]:
        return tuple(point.one_way_turnover for point in self.rebalance_points)

    def as_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.as_dict(),
            "trades": [trade.as_dict() for trade in self.trades],
            "equity_curve": list(self.equity_curve),
            "equity_points": [asdict(point) for point in self.equity_points],
            "turnover_series": list(self.turnover_series),
            "rebalance_points": [asdict(point) for point in self.rebalance_points],
            "data_quality": asdict(self.data_quality),
            "rejected_orders": self.rejected_orders,
            "total_explicit_cost": self.total_explicit_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "same_trade_gross_return": self.same_trade_gross_return,
            "average_top_n_overlap": self.average_top_n_overlap,
        }
