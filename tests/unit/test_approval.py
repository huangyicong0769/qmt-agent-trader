import pytest

from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.persistence.errors import StorageConflictError
from qmt_agent_trader.strategy.approval import (
    StrategyApproval,
    read_approval_file,
    transition_status,
    write_approval_file,
)


def test_strategy_approval_state_machine() -> None:
    assert (
        transition_status(ApprovalStatus.REVIEW_REQUIRED, ApprovalStatus.APPROVED)
        == ApprovalStatus.APPROVED
    )
    with pytest.raises(ValueError):
        transition_status(ApprovalStatus.GENERATED_BY_LLM, ApprovalStatus.APPROVED)


def test_approval_yaml_is_create_only_hashed_and_preserves_human_fields(tmp_path) -> None:
    approval = StrategyApproval(
        strategy_id="strategy_1",
        strategy_name="Strategy One",
        strategy_version="1.2.3",
        approved_by="human-reviewer",
        approved_at="2026-07-11T10:00:00+08:00",
        allowed_universe=["A_SHARE_STOCK"],
        allowed_accounts=["paper_account"],
        max_single_position_pct=0.1,
        max_turnover_daily_pct=0.2,
        max_order_value=100_000,
    )

    path = write_approval_file(approval, tmp_path)
    loaded = read_approval_file(path)

    assert loaded.approved_by == "human-reviewer"
    assert loaded.approved_at == "2026-07-11T10:00:00+08:00"
    assert len(list((tmp_path / ".manifests").glob("*.json"))) == 1
    with pytest.raises(StorageConflictError):
        write_approval_file(approval.model_copy(update={"approved_by": "agent"}), tmp_path)
    assert read_approval_file(path).approved_by == "human-reviewer"
