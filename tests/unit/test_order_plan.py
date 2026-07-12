import pytest
from pydantic import ValidationError

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.broker.order_plan import OrderPlan, OrderPlanApproval, RiskChecks
from qmt_agent_trader.core.types import ApprovalStatus, OrderType, Side
from qmt_agent_trader.persistence.artifacts import artifact_store_for_root
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageCorruptError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.services.order_plan_service import (
    OrderPlanEvent,
    verify_order_plan_event_stream,
)
from qmt_agent_trader.services.order_plan_service import (
    append_order_plan_event as _append_order_plan_event,
)
from qmt_agent_trader.services.order_plan_service import (
    load_order_plan as _load_order_plan,
)
from qmt_agent_trader.services.order_plan_service import (
    load_order_plan_events as _load_order_plan_events,
)
from qmt_agent_trader.services.order_plan_service import (
    save_order_plan as _save_order_plan,
)


def _store(root):
    return artifact_store_for_root(root, lock_manager=LockManager(root / ".test-locks"))


def save_order_plan(plan, directory):
    return _save_order_plan(plan, directory, artifact_store=_store(directory))


def load_order_plan(identifier, directory=None):
    root = directory or __import__("pathlib").Path(identifier).parent
    return _load_order_plan(identifier, root, artifact_store=_store(root))


def append_order_plan_event(order_plan_id, *, directory, **kwargs):
    return _append_order_plan_event(
        order_plan_id, directory=directory, artifact_store=_store(directory), **kwargs
    )


def load_order_plan_events(order_plan_id, directory):
    return _load_order_plan_events(
        order_plan_id, directory, artifact_store=_store(directory)
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


def test_load_rejects_unmanifested_order_plan_without_changing_bytes(tmp_path) -> None:
    plan = make_plan()
    path = tmp_path / f"{plan.order_plan_id}.json"
    original = plan.model_dump_json(indent=2).encode("utf-8")
    path.write_bytes(original)

    with pytest.raises(StorageValidationError, match="manifest is missing"):
        load_order_plan(plan.order_plan_id, tmp_path)

    assert path.read_bytes() == original
    assert not (tmp_path / ".manifests").exists()


def test_invalid_legacy_order_plan_is_structured_and_not_adopted(tmp_path) -> None:
    path = tmp_path / "op_broken.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(StorageValidationError, match="manifest is missing"):
        load_order_plan("op_broken", tmp_path)

    assert not (tmp_path / ".manifests").exists()


def test_order_plan_events_have_schema_identity_and_locked_concurrent_reads(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    plan = make_plan()
    save_order_plan(plan, tmp_path)

    def append(index: int) -> None:
        append_order_plan_event(
            plan.order_plan_id,
            directory=tmp_path,
            event_type="RISK_CHECKED",
            actor="test",
            details={"index": index},
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(12)))
        snapshots = list(
            executor.map(lambda _: load_order_plan_events(plan.order_plan_id, tmp_path), range(4))
        )

    assert all(len(snapshot) == 12 for snapshot in snapshots)
    events = snapshots[0]
    assert all(event.schema_version == 1 and event.event_id for event in events)
    assert {event.details["index"] for event in events} == set(range(12))


def test_order_plan_event_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValidationError):
        OrderPlanEvent(
            schema_version=2,
            order_plan_id="op_1",
            event_type="RISK_CHECKED",
            actor="test",
        )


def test_order_plan_event_truncated_tail_fails_closed(tmp_path) -> None:
    plan = make_plan()
    save_order_plan(plan, tmp_path)
    append_order_plan_event(
        plan.order_plan_id,
        directory=tmp_path,
        event_type="RISK_CHECKED",
        actor="test",
    )
    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    event_path.write_bytes(event_path.read_bytes() + b'{"broken"')

    with pytest.raises(StorageCorruptError, match="truncated tail"):
        load_order_plan_events(plan.order_plan_id, tmp_path)
    with pytest.raises(StorageCorruptError, match="cannot append"):
        append_order_plan_event(
            plan.order_plan_id,
            directory=tmp_path,
            event_type="PAPER_ACCEPTED",
            actor="test",
        )

    verification = verify_order_plan_event_stream(
        event_path, expected_order_plan_id=plan.order_plan_id
    )
    assert not verification.healthy
    assert verification.tail_truncated


def test_order_plan_event_verifier_rejects_duplicate_event_identity(tmp_path) -> None:
    plan = make_plan()
    save_order_plan(plan, tmp_path)
    append_order_plan_event(
        plan.order_plan_id,
        directory=tmp_path,
        event_type="RISK_CHECKED",
        actor="test",
    )
    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    event_path.write_bytes(event_path.read_bytes() * 2)

    verification = verify_order_plan_event_stream(
        event_path, expected_order_plan_id=plan.order_plan_id
    )
    assert not verification.healthy
    assert verification.event_count == 2
    assert any(item.code == "DUPLICATE_EVENT_ID" for item in verification.corruptions)


def test_order_plan_event_verifier_rejects_filename_binding_mismatch(tmp_path) -> None:
    plan = make_plan()
    save_order_plan(plan, tmp_path)
    append_order_plan_event(
        plan.order_plan_id,
        directory=tmp_path,
        event_type="RISK_CHECKED",
        actor="test",
    )
    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    mismatched = event_path.with_name("0" * 64 + ".jsonl")
    event_path.rename(mismatched)

    verification = verify_order_plan_event_stream(mismatched)
    assert not verification.healthy
    assert any(item.code == "FILENAME_ID_MISMATCH" for item in verification.corruptions)
