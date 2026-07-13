import hashlib
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.broker.order_plan import OrderPlan, OrderPlanApproval, RiskChecks
from qmt_agent_trader.core.types import ApprovalStatus, OrderType, Side
from qmt_agent_trader.persistence.artifacts import ArtifactMetadata, artifact_store_for_root
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageCorruptError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.services.order_plan_service import (
    OrderPlanEvent,
    append_order_plan_event,
    load_order_plan,
    load_order_plan_events,
    save_order_plan,
    verify_order_plan_event_stream,
)


def _store(root):
    return artifact_store_for_root(root, lock_manager=LockManager(root / ".test-locks"))


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
    store = _store(tmp_path)
    path = save_order_plan(plan, artifact_store=store)
    loaded = load_order_plan(path.as_posix(), artifact_store=store)

    assert loaded.order_plan_id == plan.order_plan_id
    assert loaded.plan_hash == plan.plan_hash
    assert len(list((tmp_path / ".manifests").glob("*.json"))) == 1


def test_load_order_plan_accepts_relative_filename(tmp_path: Path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    path = save_order_plan(plan, artifact_store=store)

    loaded = load_order_plan(path.name, artifact_store=store)

    assert loaded.order_plan_id == plan.order_plan_id


@pytest.mark.parametrize("identifier", ["plan.yaml", "", "."])
def test_load_order_plan_rejects_invalid_identifier(
    tmp_path: Path,
    identifier: str,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(StorageValidationError, match="identifier"):
        load_order_plan(identifier, artifact_store=store)


def test_load_order_plan_rejects_relative_identifier_with_directory_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    path = save_order_plan(plan, artifact_store=store)
    monkeypatch.chdir(store.root)

    with pytest.raises(StorageValidationError, match="identifier"):
        load_order_plan(f"nested/../{path.name}", artifact_store=store)


def test_load_order_plan_identity_mismatch_is_structured(tmp_path: Path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    alias = "op_alias"
    store.create(
        f"{alias}.json",
        plan.model_dump_json(indent=2).encode("utf-8"),
        metadata=ArtifactMetadata(
            artifact_id=alias,
            artifact_type="order_plan",
            producer="tests.unit.test_order_plan",
        ),
    )

    with pytest.raises(StorageValidationError, match="does not match repository path"):
        load_order_plan(alias, artifact_store=store)


def test_order_plan_is_create_only_and_verified_before_execution(tmp_path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    path = save_order_plan(plan, artifact_store=store)

    with pytest.raises(StorageConflictError):
        save_order_plan(plan, artifact_store=store)
    path.write_text(path.read_text(encoding="utf-8").replace("acct", "evil"), encoding="utf-8")
    with pytest.raises(StorageValidationError, match="hash_mismatch"):
        load_order_plan(plan.order_plan_id, artifact_store=store)


def test_order_plan_events_append_without_mutating_original_plan(tmp_path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    path = save_order_plan(plan, artifact_store=store)
    original = path.read_bytes()

    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="cli",
        details={"status": "PASSED"},
        artifact_store=store,
    )
    append_order_plan_event(
        plan.order_plan_id,
        event_type="PAPER_ACCEPTED",
        actor="cli",
        details={"live": False},
        artifact_store=store,
    )

    loaded = load_order_plan(plan.order_plan_id, artifact_store=store)
    events = load_order_plan_events(plan.order_plan_id, artifact_store=store)
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
        load_order_plan(plan.order_plan_id, artifact_store=_store(tmp_path))

    assert path.read_bytes() == original
    assert not (tmp_path / ".manifests").exists()


def test_invalid_legacy_order_plan_is_structured_and_not_adopted(tmp_path) -> None:
    path = tmp_path / "op_broken.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(StorageValidationError, match="manifest is missing"):
        load_order_plan("op_broken", artifact_store=_store(tmp_path))

    assert not (tmp_path / ".manifests").exists()


def test_order_plan_path_cannot_select_foreign_artifact_root(tmp_path) -> None:
    foreign = tmp_path / "foreign/op.json"
    foreign.parent.mkdir()
    foreign.write_text("{}")

    with pytest.raises(StorageValidationError, match="outside the artifact root"):
        load_order_plan(str(foreign), artifact_store=_store(tmp_path / "plans"))


def test_order_plan_events_have_schema_identity_and_locked_concurrent_reads(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    plan = make_plan()
    store = _store(tmp_path)
    save_order_plan(plan, artifact_store=store)

    def append(index: int) -> None:
        append_order_plan_event(
            plan.order_plan_id,
            event_type="RISK_CHECKED",
            actor="test",
            details={"index": index},
            artifact_store=store,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(12)))
        snapshots = list(
            executor.map(
                lambda _: load_order_plan_events(
                    plan.order_plan_id, artifact_store=store
                ),
                range(4),
            )
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
    store = _store(tmp_path)
    save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
    )
    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    event_path.write_bytes(event_path.read_bytes() + b'{"broken"')

    with pytest.raises(StorageCorruptError, match="truncated tail"):
        load_order_plan_events(plan.order_plan_id, artifact_store=store)
    with pytest.raises(StorageCorruptError, match="cannot append"):
        append_order_plan_event(
            plan.order_plan_id,
            event_type="PAPER_ACCEPTED",
            actor="test",
            artifact_store=store,
        )

    verification = verify_order_plan_event_stream(
        event_path, expected_order_plan_id=plan.order_plan_id
    )
    assert not verification.healthy
    assert verification.tail_truncated


def test_existing_empty_order_plan_event_stream_fails_closed(tmp_path: Path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    save_order_plan(plan, artifact_store=store)
    event_dir = tmp_path / ".events"
    event_dir.mkdir()
    event_path = event_dir / f"{hashlib.sha256(plan.order_plan_id.encode()).hexdigest()}.jsonl"
    event_path.touch()

    with pytest.raises(StorageCorruptError, match="ORPHAN_EVENT_STREAM"):
        load_order_plan_events(plan.order_plan_id, artifact_store=store)


def test_event_stream_without_order_plan_is_rejected(tmp_path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    content = save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
    )

    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    content.unlink()
    store.manifest_path_for(plan.order_plan_id).unlink()

    with pytest.raises(StorageCorruptError, match="order plan"):
        load_order_plan_events(plan.order_plan_id, artifact_store=store)

    assert event_path.exists()


def test_event_stream_with_tampered_order_plan_is_rejected(tmp_path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    content = save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
    )

    content.write_bytes(b"tampered")

    with pytest.raises(StorageCorruptError, match="order plan"):
        load_order_plan_events(plan.order_plan_id, artifact_store=store)


def test_order_plan_event_verifier_rejects_duplicate_event_identity(tmp_path) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
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
    store = _store(tmp_path)
    save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
    )
    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    mismatched = event_path.with_name("0" * 64 + ".jsonl")
    event_path.rename(mismatched)

    verification = verify_order_plan_event_stream(mismatched)
    assert not verification.healthy
    assert any(item.code == "FILENAME_ID_MISMATCH" for item in verification.corruptions)


def test_event_append_cannot_recreate_stream_after_complete_quarantine(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    content = save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
    )
    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    content.write_bytes(b"tampered")
    entered = threading.Event()
    release = threading.Event()
    append_attempting = threading.Event()
    append_done = threading.Event()
    append_errors: list[Exception] = []
    original = store._quarantine_assume_locked

    def paused_quarantine(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(**kwargs)

    monkeypatch.setattr(store, "_quarantine_assume_locked", paused_quarantine)
    quarantine_thread = threading.Thread(
        target=lambda: store.quarantine(
            artifact_id=plan.order_plan_id,
            expected_relative_path=f"{plan.order_plan_id}.json",
            quarantine_root=tmp_path / "quarantine",
            auxiliary_paths=(event_path,),
        )
    )
    quarantine_thread.start()
    assert entered.wait(timeout=5)

    def append_after_quarantine() -> None:
        append_attempting.set()
        try:
            append_order_plan_event(
                plan.order_plan_id,
                event_type="PAPER_ACCEPTED",
                actor="test",
                artifact_store=store,
            )
        except Exception as exc:
            append_errors.append(exc)
        finally:
            append_done.set()

    append_thread = threading.Thread(target=append_after_quarantine)
    append_thread.start()
    assert append_attempting.wait(timeout=5)
    assert not append_done.is_set()
    release.set()
    quarantine_thread.join(timeout=5)
    append_thread.join(timeout=5)

    assert append_done.is_set()
    assert append_errors and isinstance(append_errors[0], StorageValidationError)
    assert not event_path.exists()


def test_event_read_waits_for_complete_quarantine(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = make_plan()
    store = _store(tmp_path)
    content = save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
    )
    event_path = next((tmp_path / ".events").glob("*.jsonl"))
    content.write_bytes(b"tampered")
    entered = threading.Event()
    release = threading.Event()
    read_attempting = threading.Event()
    read_done = threading.Event()
    read_result: list[list[OrderPlanEvent]] = []
    original = store._quarantine_assume_locked

    def paused_quarantine(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(**kwargs)

    monkeypatch.setattr(store, "_quarantine_assume_locked", paused_quarantine)
    quarantine_thread = threading.Thread(
        target=lambda: store.quarantine(
            artifact_id=plan.order_plan_id,
            expected_relative_path=f"{plan.order_plan_id}.json",
            quarantine_root=tmp_path / "quarantine",
            auxiliary_paths=(event_path,),
        )
    )
    quarantine_thread.start()
    assert entered.wait(timeout=5)

    def read_during_quarantine() -> None:
        read_attempting.set()
        read_result.append(
            load_order_plan_events(plan.order_plan_id, artifact_store=store)
        )
        read_done.set()

    read_thread = threading.Thread(target=read_during_quarantine)
    read_thread.start()
    assert read_attempting.wait(timeout=5)
    assert not read_done.is_set()
    release.set()
    quarantine_thread.join(timeout=5)
    read_thread.join(timeout=5)

    assert read_done.is_set()
    assert read_result == [[]]
