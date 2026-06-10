"""Sensitivity analysis primitives for strategy robustness research."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product
from statistics import median
from typing import Protocol


@dataclass(frozen=True)
class SensitivityScenario:
    cost_multiplier: float = 1.0
    slippage_bps: float = 0.0
    execution_delay_days: int = 1
    top_n: int | None = None
    max_single_position_pct: float | None = None

    def label(self) -> str:
        fields = [
            f"cost_x{self.cost_multiplier:g}",
            f"slip_{self.slippage_bps:g}bps",
            f"delay_{self.execution_delay_days}d",
        ]
        if self.top_n is not None:
            fields.append(f"top_{self.top_n}")
        if self.max_single_position_pct is not None:
            fields.append(f"maxpos_{self.max_single_position_pct:g}")
        return "__".join(fields)


@dataclass(frozen=True)
class SensitivityGrid:
    cost_multipliers: tuple[float, ...] = (1.0,)
    slippage_bps: tuple[float, ...] = (0.0,)
    execution_delay_days: tuple[int, ...] = (1,)
    top_n: tuple[int | None, ...] = (None,)
    max_single_position_pct: tuple[float | None, ...] = (None,)

    def scenarios(self) -> list[SensitivityScenario]:
        self._validate()
        return [
            SensitivityScenario(
                cost_multiplier=cost_multiplier,
                slippage_bps=slippage,
                execution_delay_days=delay,
                top_n=top_n,
                max_single_position_pct=max_position,
            )
            for cost_multiplier, slippage, delay, top_n, max_position in product(
                self.cost_multipliers,
                self.slippage_bps,
                self.execution_delay_days,
                self.top_n,
                self.max_single_position_pct,
            )
        ]

    def _validate(self) -> None:
        if not self.cost_multipliers:
            raise ValueError("cost_multipliers cannot be empty")
        if not self.slippage_bps:
            raise ValueError("slippage_bps cannot be empty")
        if not self.execution_delay_days:
            raise ValueError("execution_delay_days cannot be empty")
        if any(value <= 0 for value in self.cost_multipliers):
            raise ValueError("cost multipliers must be positive")
        if any(value < 0 for value in self.slippage_bps):
            raise ValueError("slippage bps values cannot be negative")
        if any(value < 0 for value in self.execution_delay_days):
            raise ValueError("execution delay days cannot be negative")
        if any(value is not None and value <= 0 for value in self.top_n):
            raise ValueError("top_n values must be positive when provided")
        if any(
            value is not None and (value <= 0 or value > 1)
            for value in self.max_single_position_pct
        ):
            raise ValueError("max position pct must be in (0, 1] when provided")


@dataclass(frozen=True)
class SensitivityMetrics:
    total_return: float
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    diagnostic_pass: bool = True

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "diagnostic_pass": self.diagnostic_pass,
        }


@dataclass(frozen=True)
class SensitivityRun:
    scenario: SensitivityScenario
    metrics: SensitivityMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario.__dict__,
            "label": self.scenario.label(),
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class SensitivitySummary:
    scenario_count: int
    baseline_total_return: float
    worst_total_return: float
    median_total_return: float
    return_degradation: float
    pass_ratio: float
    worst_scenario: SensitivityScenario | None

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_count": self.scenario_count,
            "baseline_total_return": self.baseline_total_return,
            "worst_total_return": self.worst_total_return,
            "median_total_return": self.median_total_return,
            "return_degradation": self.return_degradation,
            "pass_ratio": self.pass_ratio,
            "worst_scenario": (
                None if self.worst_scenario is None else self.worst_scenario.__dict__
            ),
        }


@dataclass(frozen=True)
class SensitivityReport:
    runs: tuple[SensitivityRun, ...]
    summary: SensitivitySummary

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary.as_dict(),
            "runs": [run.as_dict() for run in self.runs],
        }


class SensitivityRunner(Protocol):
    def __call__(self, scenario: SensitivityScenario) -> SensitivityMetrics:
        """Run one scenario and return realized metrics."""


class SensitivityAnalyzer:
    def run(
        self,
        scenarios: Iterable[SensitivityScenario],
        runner: SensitivityRunner,
        *,
        baseline_selector: Callable[[SensitivityScenario], bool] | None = None,
    ) -> SensitivityReport:
        runs = tuple(SensitivityRun(scenario, runner(scenario)) for scenario in scenarios)
        return SensitivityReport(
            runs=runs,
            summary=self._summarize(runs, baseline_selector=baseline_selector),
        )

    def _summarize(
        self,
        runs: tuple[SensitivityRun, ...],
        *,
        baseline_selector: Callable[[SensitivityScenario], bool] | None,
    ) -> SensitivitySummary:
        if not runs:
            return SensitivitySummary(
                scenario_count=0,
                baseline_total_return=0.0,
                worst_total_return=0.0,
                median_total_return=0.0,
                return_degradation=0.0,
                pass_ratio=0.0,
                worst_scenario=None,
            )
        baseline = self._choose_baseline(runs, baseline_selector)
        worst = min(runs, key=lambda run: run.metrics.total_return)
        returns = [run.metrics.total_return for run in runs]
        return SensitivitySummary(
            scenario_count=len(runs),
            baseline_total_return=baseline.metrics.total_return,
            worst_total_return=worst.metrics.total_return,
            median_total_return=float(median(returns)),
            return_degradation=baseline.metrics.total_return - worst.metrics.total_return,
            pass_ratio=sum(1 for run in runs if run.metrics.diagnostic_pass) / len(runs),
            worst_scenario=worst.scenario,
        )

    @staticmethod
    def _choose_baseline(
        runs: tuple[SensitivityRun, ...],
        selector: Callable[[SensitivityScenario], bool] | None,
    ) -> SensitivityRun:
        if selector is not None:
            for run in runs:
                if selector(run.scenario):
                    return run
        for run in runs:
            scenario = run.scenario
            if (
                scenario.cost_multiplier == 1.0
                and scenario.slippage_bps == 0.0
                and scenario.execution_delay_days == 1
            ):
                return run
        return runs[0]

