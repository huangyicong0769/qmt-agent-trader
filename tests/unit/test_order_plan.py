import pytest

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.broker.order_plan import OrderPlan, OrderPlanApproval, RiskChecks
from qmt_agent_trader.core.types import ApprovalStatus, OrderType, Side
from qmt_agent_trader.persistence.errors import StorageConflictError, StorageValidationError
from qmt_agent_trader.services.order_plan_service import (
    append_order_plan_event,
    load_order_plan,
    load_order_plan_events,
    save_order_plan,
)


def make_plan(status: ApprovalStatus = ApprovalStatus.APPROVED) -> OrderPlan:
    return OrderPlan(
        strategy_id="s1",
        strategy_version="1.0.0",
        strategy_approval_status=status,
        account_id_hash="acct",
        dry_run=True,
        orders=(
            Order(
                symbol="000001.SZ",
                side=Side.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=10.0,
                reason="test",
            ),
        ),
        risk_checks=RiskChecks.passed(["cash_available"]),
        approval=OrderPlanApproval(status=ApprovalStatus.APPROVED),
    )


def test_order_plan_hash_and_idempotency_key() -> None:
    plan = make_plan()
    assert plan.plan_hash == plan.compute_hash()
    assert plan.idempotency_key


def test_unapproved_strategy_cannot_submit() -> None:
    plan = make_plan(status=ApprovalStatus.REVIEW_REQUIRED)
    with pytest.raises(ValueError, match="strategy is not approved"):
        plan.assert_submittable()


def test_order_plan_save_and_load(tmp_path) -> None:
    plan = make_plan()
    path = save_order_plan(plan, tmp_path)
    loaded = load_order_plan(path.as_posix())

    assert loaded.order_plan_id == plan.order_plan_id
    assert loaded.plan_hash == plan.plan_hash
    assert len(list((tmp_path / ".manifests").glob("*.json"))) == 1


def test_order_plan_is_create_only_and_verified_before_execution(tmp_path) -> None:
    plan = make_plan()
    path = save_order_plan(plan, tmp_path)

    with pytest.raises(StorageConflictError):
        save_order_plan(plan, tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("acct", "evil"), encoding="utf-8")
    with pytest.raises(StorageValidationError, match="hash_mismatch"):
        load_order_plan(plan.order_plan_id, tmp_path)


def test_order_plan_events_append_without_mutating_original_plan(tmp_path) -> None:
    plan = make_plan()
    path = save_order_plan(plan, tmp_path)
    original = path.read_bytes()

    append_order_plan_event(
        plan.order_plan_id,
        directory=tmp_path,
        event_type="RISK_CHECKED",
        actor="cli",
        details={"status": "PASSED"},
    )
    append_order_plan_event(
        plan.order_plan_id,
        directory=tmp_path,
        event_type="PAPER_ACCEPTED",
        actor="cli",
        details={"live": False},
    )

    loaded = load_order_plan(plan.order_plan_id, tmp_path)
    events = load_order_plan_events(plan.order_plan_id, tmp_path)
    assert loaded.plan_hash == plan.plan_hash
    assert path.read_bytes() == original
    assert [event.event_type for event in events] == ["RISK_CHECKED", "PAPER_ACCEPTED"]
    assert all(event.order_plan_id == plan.order_plan_id for event in events)
