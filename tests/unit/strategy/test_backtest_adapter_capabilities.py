import pytest

from qmt_agent_trader.strategy.adapter_capabilities import validate_factor_rank_adapter_spec
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


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"portfolio": {"method": "score_weighted_top_n"}}, "portfolio.method"),
        ({"execution": {"execution_timing": "same_close"}}, "execution.execution_timing"),
        ({"kind": "CUSTOM"}, "kind"),
    ],
)
def test_unsupported_semantics_are_reported(update, field) -> None:
    payload = _base_spec().model_dump(mode="json")
    payload.update(update)
    issues = validate_factor_rank_adapter_spec(StrategySpec.model_validate(payload))
    assert field in {issue.field for issue in issues}


def test_code_path_is_never_silently_ignored() -> None:
    issues = validate_factor_rank_adapter_spec(_base_spec(), code_path="generated/strategy.py")
    assert [issue.field for issue in issues] == ["code_path"]
