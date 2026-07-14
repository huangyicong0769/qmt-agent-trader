"""Fail-closed capability contract for the canonical factor-rank adapter."""

from __future__ import annotations

from dataclasses import dataclass

from qmt_agent_trader.strategy.models import StrategyKind, StrategySpec

CANONICAL_FACTOR_RANK_SEMANTIC_FIELDS = {
    "kind",
    "portfolio.method",
    "portfolio.top_n",
    "portfolio.max_single_position_pct",
    "portfolio.cash_buffer_pct",
    "portfolio.long_only",
    "rebalance.frequency",
    "rebalance.min_turnover_threshold",
    "rebalance.rank_buffer",
    "execution.signal_timing",
    "execution.execution_timing",
    "execution.execution_delay_days",
    "execution.slippage_bps",
    "execution.cost_model",
    "factors[].ascending",
    "factors[].weight",
    "factors[].transform",
    "risk_constraints",
}


@dataclass(frozen=True)
class AdapterCapabilityIssue:
    field: str
    observed: object
    supported: object
    message: str


def validate_factor_rank_adapter_spec(
    spec: StrategySpec,
    code_path: str | None = None,
) -> tuple[AdapterCapabilityIssue, ...]:
    checks: list[tuple[str, object, object, bool]] = [
        ("kind", spec.kind.value, StrategyKind.FACTOR_RANK_LONG_ONLY.value,
         spec.kind == StrategyKind.FACTOR_RANK_LONG_ONLY),
        ("portfolio.method", spec.portfolio.method, "equal_weight_top_n",
         spec.portfolio.method == "equal_weight_top_n"),
        ("portfolio.long_only", spec.portfolio.long_only, True, spec.portfolio.long_only is True),
        ("rebalance.frequency", spec.rebalance.frequency, ("daily", "weekly", "monthly"),
         spec.rebalance.frequency in {"daily", "weekly", "monthly"}),
        ("execution.signal_timing", spec.execution.signal_timing, "after_close",
         spec.execution.signal_timing == "after_close"),
        ("execution.execution_timing", spec.execution.execution_timing, "next_open",
         spec.execution.execution_timing == "next_open"),
        ("execution.execution_delay_days", spec.execution.execution_delay_days, ">=1",
         spec.execution.execution_delay_days >= 1),
        (
            "execution.cost_model",
            spec.execution.cost_model,
            "a_share_default",
            spec.execution.cost_model == "a_share_default",
        ),
    ]
    issues = [
        AdapterCapabilityIssue(
            field=field,
            observed=observed,
            supported=supported,
            message=f"factor-rank adapter does not support {field}={observed!r}",
        )
        for field, observed, supported, valid in checks
        if not valid
    ]
    for index, factor in enumerate(spec.factors):
        if factor.transform is not None:
            issues.append(
                AdapterCapabilityIssue(
                    field=f"factors[{index}].transform",
                    observed=factor.transform,
                    supported=None,
                    message="factor transforms are not implemented by the canonical adapter",
                )
            )
    if spec.risk_constraints:
        issues.append(
            AdapterCapabilityIssue(
                field="risk_constraints",
                observed=dict(spec.risk_constraints),
                supported={},
                message="canonical factor-rank adapter does not execute risk_constraints",
            )
        )
    if code_path:
        issues.append(
            AdapterCapabilityIssue(
                field="code_path",
                observed=code_path,
                supported=None,
                message="generated strategy execution requires a process-isolated runner",
            )
        )
    return tuple(issues)
