"""Research diagnostics for candidate strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from statistics import mean
from typing import Any


class DiagnosticStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NOT_COMPUTED = "NOT_COMPUTED"


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    status: DiagnosticStatus
    observed: float | bool | str
    threshold: float | bool | str
    message: str
    evidence_source: str = "computed"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "message": self.message,
            "evidence_source": self.evidence_source,
        }


@dataclass(frozen=True)
class StrategyDiagnosticConfig:
    min_observations: int = 252
    min_trade_count: int = 10
    min_coverage: float = 0.80
    min_positive_ic_ratio: float = 0.52
    min_walk_forward_positive_ratio: float = 0.50
    max_abs_drawdown: float = 0.25
    max_average_turnover: float = 0.40
    max_cost_to_initial_cash: float = 0.02
    max_rejection_rate: float = 0.10


@dataclass(frozen=True)
class StrategyDiagnostics:
    status: DiagnosticStatus
    checks: tuple[DiagnosticCheck, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "checks": [check.as_dict() for check in self.checks],
        }


class StrategyDiagnosticsEvaluator:
    """Evaluate evidence quality without approving or rejecting a strategy."""

    def evaluate(
        self,
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig | None = None,
    ) -> StrategyDiagnostics:
        cfg = StrategyDiagnosticConfig() if config is None else config
        checks = (
            self._leakage_check(evidence),
            self._sample_size_check(evidence, cfg),
            self._trade_count_check(evidence, cfg),
            self._coverage_check(evidence, cfg),
            self._daily_cross_section_check(evidence),
            self._positive_ic_check(evidence, cfg),
            self._walk_forward_check(evidence, cfg),
            self._drawdown_check(evidence, cfg),
            self._turnover_check(evidence, cfg),
            self._cost_check(evidence, cfg),
            self._rejection_rate_check(evidence, cfg),
        )
        return StrategyDiagnostics(status=self._overall_status(checks), checks=checks)

    @staticmethod
    def _overall_status(checks: tuple[DiagnosticCheck, ...]) -> DiagnosticStatus:
        if any(check.status == DiagnosticStatus.FAIL for check in checks):
            return DiagnosticStatus.FAIL
        if any(
            check.status in {DiagnosticStatus.WARN, DiagnosticStatus.NOT_COMPUTED}
            for check in checks
        ):
            return DiagnosticStatus.WARN
        return DiagnosticStatus.PASS

    @staticmethod
    def _leakage_check(evidence: dict[str, Any]) -> DiagnosticCheck:
        observed = bool(_dig(evidence, "leakage_report", "valid", default=False))
        return DiagnosticCheck(
            name="leakage_valid",
            status=DiagnosticStatus.PASS if observed else DiagnosticStatus.FAIL,
            observed=observed,
            threshold=True,
            message="leakage report must be valid",
        )

    @staticmethod
    def _sample_size_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        observed = _float_metric(evidence, "factor_report", "observation_count")
        status = (
            DiagnosticStatus.PASS
            if observed >= config.min_observations
            else DiagnosticStatus.WARN
        )
        return DiagnosticCheck(
            name="min_observations",
            status=status,
            observed=observed,
            threshold=float(config.min_observations),
            message="factor sample should be large enough for non-experimental research",
        )

    @staticmethod
    def _trade_count_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        trade_blotter = _dig(evidence, "trade_blotter", default={})
        observed = (
            float(len(trade_blotter))
            if isinstance(trade_blotter, list)
            else _float_metric(evidence, "trade_blotter", "count")
        )
        status = (
            DiagnosticStatus.PASS
            if observed >= config.min_trade_count
            else DiagnosticStatus.WARN
        )
        return DiagnosticCheck(
            name="min_trade_count",
            status=status,
            observed=observed,
            threshold=float(config.min_trade_count),
            message="strategy needs enough simulated trades to support review",
        )

    @staticmethod
    def _coverage_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        if not _has_metric(evidence, "factor_report", "coverage"):
            return _not_computed_check(
                "coverage",
                config.min_coverage,
                "factor coverage was not computed; do not infer coverage quality",
            )
        observed = _float_metric(evidence, "factor_report", "coverage")
        status = (
            DiagnosticStatus.PASS
            if observed >= config.min_coverage
            else DiagnosticStatus.WARN
        )
        return DiagnosticCheck(
            name="coverage",
            status=status,
            observed=observed,
            threshold=config.min_coverage,
            message="factor should cover most eligible symbol-date observations",
        )

    @staticmethod
    def _positive_ic_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        if not _has_metric(evidence, "factor_report", "positive_ic_ratio"):
            return _not_computed_check(
                "positive_ic_ratio",
                config.min_positive_ic_ratio,
                "positive IC ratio was not computed; do not infer predictive quality",
            )
        observed = _float_metric(evidence, "factor_report", "positive_ic_ratio")
        status = (
            DiagnosticStatus.PASS
            if observed >= config.min_positive_ic_ratio
            else DiagnosticStatus.WARN
        )
        return DiagnosticCheck(
            name="positive_ic_ratio",
            status=status,
            observed=observed,
            threshold=config.min_positive_ic_ratio,
            message="daily IC should be positive in enough periods",
        )

    @staticmethod
    def _daily_cross_section_check(evidence: dict[str, Any]) -> DiagnosticCheck:
        dates = _dig(evidence, "data_quality", "abrupt_low_coverage_dates", default=[])
        count = len(dates) if isinstance(dates, list | tuple) else 0
        return DiagnosticCheck(
            name="daily_cross_sectional_coverage",
            status=DiagnosticStatus.PASS if count == 0 else DiagnosticStatus.FAIL,
            observed=count,
            threshold=0,
            message="completed backtests must have no abrupt daily coverage collapse",
            evidence_source="input_panel_metadata",
        )
    @staticmethod
    def _walk_forward_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        slices = _dig(evidence, "factor_report", "walk_forward", default=[])
        if not isinstance(slices, list):
            slices = []
        if not slices:
            return _not_computed_check(
                "walk_forward_consistency",
                config.min_walk_forward_positive_ratio,
                "walk-forward factor direction was not computed",
            )
        observed = _positive_walk_forward_ratio(slices)
        return DiagnosticCheck(
            name="walk_forward_consistency",
            status=(
                DiagnosticStatus.PASS
                if observed >= config.min_walk_forward_positive_ratio
                else DiagnosticStatus.WARN
            ),
            observed=observed,
            threshold=config.min_walk_forward_positive_ratio,
            message="walk-forward slices should show consistent factor direction",
        )

    @staticmethod
    def _drawdown_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        observed = abs(_float_metric(evidence, "performance_report", "max_drawdown"))
        status = (
            DiagnosticStatus.PASS
            if observed <= config.max_abs_drawdown
            else DiagnosticStatus.FAIL
        )
        return DiagnosticCheck(
            name="max_drawdown",
            status=status,
            observed=observed,
            threshold=config.max_abs_drawdown,
            message="portfolio drawdown should stay within configured research risk tolerance",
        )

    @staticmethod
    def _turnover_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        observed = _average_turnover(_dig(evidence, "turnover_report", default={}))
        status = (
            DiagnosticStatus.PASS
            if observed <= config.max_average_turnover
            else DiagnosticStatus.WARN
        )
        return DiagnosticCheck(
            name="average_turnover",
            status=status,
            observed=observed,
            threshold=config.max_average_turnover,
            message="turnover should be low enough for daily execution after costs",
        )

    @staticmethod
    def _cost_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        observed = _float_metric(evidence, "cost_report", "cost_to_initial_cash")
        status = (
            DiagnosticStatus.PASS
            if observed <= config.max_cost_to_initial_cash
            else DiagnosticStatus.WARN
        )
        return DiagnosticCheck(
            name="cost_to_initial_cash",
            status=status,
            observed=observed,
            threshold=config.max_cost_to_initial_cash,
            message="trading costs should not dominate research-period capital",
        )

    @staticmethod
    def _rejection_rate_check(
        evidence: dict[str, Any],
        config: StrategyDiagnosticConfig,
    ) -> DiagnosticCheck:
        observed = _float_metric(evidence, "rejection_report", "rate")
        status = (
            DiagnosticStatus.PASS
            if observed <= config.max_rejection_rate
            else DiagnosticStatus.WARN
        )
        return DiagnosticCheck(
            name="rejection_rate",
            status=status,
            observed=observed,
            threshold=config.max_rejection_rate,
            message="too many simulated orders rejected by trading constraints",
        )


def _dig(evidence: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = evidence
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _float_metric(evidence: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value = _dig(evidence, *keys, default=default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return default


def _has_metric(evidence: dict[str, Any], *keys: str) -> bool:
    sentinel = object()
    return _dig(evidence, *keys, default=sentinel) is not sentinel


def _not_computed_check(name: str, threshold: float | bool, message: str) -> DiagnosticCheck:
    return DiagnosticCheck(
        name=name,
        status=DiagnosticStatus.NOT_COMPUTED,
        observed="NOT_COMPUTED",
        threshold=threshold,
        message=message,
        evidence_source="not_computed",
    )


def _average_turnover(payload: Any) -> float:
    if isinstance(payload, dict):
        if isinstance(payload.get("average_turnover"), int | float):
            return float(payload["average_turnover"])
        values = payload.get("turnovers")
    else:
        values = payload
    if not isinstance(values, list) or not values:
        return 0.0
    concrete = [float(value) for value in values if isinstance(value, int | float)]
    return float(mean(concrete)) if concrete else 0.0


def _positive_walk_forward_ratio(slices: list[Any]) -> float:
    if not slices:
        return 0.0
    positive = 0
    concrete_count = 0
    for item in slices:
        if not isinstance(item, dict):
            continue
        mean_ic = item.get("mean_ic", 0.0)
        spread = item.get("long_short_spread", 0.0)
        if not isinstance(mean_ic, int | float) or not isinstance(spread, int | float):
            continue
        concrete_count += 1
        if mean_ic > 0 and spread > 0:
            positive += 1
    return positive / concrete_count if concrete_count else 0.0
