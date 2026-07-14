import pytest

from qmt_agent_trader.strategy.adapter_capabilities import (
    CANONICAL_FACTOR_RANK_SEMANTIC_FIELDS,
    validate_factor_rank_adapter_spec,
)
from qmt_agent_trader.strategy.models import StrategySpec


def _base_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": "factor_rank",
            "name": "Factor rank",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "momentum_20d"}],
        }
    )


def deep_merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"portfolio": {"method": "score_weighted_top_n"}}, "portfolio.method"),
        ({"execution": {"execution_timing": "same_close"}}, "execution.execution_timing"),
        ({"execution": {"cost_model": "zero_cost"}}, "execution.cost_model"),
        ({"risk_constraints": {"stop_loss_pct": 0.10}}, "risk_constraints"),
        ({"kind": "CUSTOM"}, "kind"),
    ],
)
def test_unsupported_semantics_are_reported(update, field) -> None:
    payload = _base_spec().model_dump(mode="json")
    payload = deep_merge(payload, update)
    issues = validate_factor_rank_adapter_spec(StrategySpec.model_validate(payload))
    assert field in {issue.field for issue in issues}


def test_code_path_is_never_silently_ignored() -> None:
    issues = validate_factor_rank_adapter_spec(_base_spec(), code_path="generated/strategy.py")
    assert [issue.field for issue in issues] == ["code_path"]


def test_capability_contract_tracks_all_declared_strategy_fields() -> None:
    declared = {
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
    assert CANONICAL_FACTOR_RANK_SEMANTIC_FIELDS == declared
