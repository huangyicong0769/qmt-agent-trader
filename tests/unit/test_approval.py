import pytest

from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.strategy.approval import transition_status


def test_strategy_approval_state_machine() -> None:
    assert (
        transition_status(ApprovalStatus.REVIEW_REQUIRED, ApprovalStatus.APPROVED)
        == ApprovalStatus.APPROVED
    )
    with pytest.raises(ValueError):
        transition_status(ApprovalStatus.GENERATED_BY_LLM, ApprovalStatus.APPROVED)
