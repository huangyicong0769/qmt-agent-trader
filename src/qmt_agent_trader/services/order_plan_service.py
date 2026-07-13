"""Order plan generation service."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.broker.order_plan import OrderPlan, OrderPlanApproval, RiskChecks
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus, OrderType, Side
from qmt_agent_trader.persistence.artifacts import (
    ArtifactMetadata,
    ArtifactStore,
)
from qmt_agent_trader.persistence.errors import StorageCorruptError, StorageValidationError


def build_sample_paper_order_plan(strategy_id: str) -> OrderPlan:
    return OrderPlan(
        strategy_id=strategy_id,
        strategy_version="1.0.0",
        strategy_approval_status=ApprovalStatus.APPROVED,
        account_id_hash="paper_account",
        dry_run=True,
        orders=(
            Order(
                symbol="000001.SZ",
                side=Side.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=10.0,
                reason="sample paper rebalance",
            ),
        ),
        risk_checks=RiskChecks.passed(["max_order_value", "cash_available"]),
        approval=OrderPlanApproval(status=ApprovalStatus.APPROVED, approved_by="human"),
    )


class OrderPlanEvent(BaseModel):
    schema_version: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: new_id("ope"))
    order_plan_id: str
    event_type: str
    actor: str
    created_at: str = Field(default_factory=shanghai_now_iso)
    details: dict[str, object] = Field(default_factory=dict)


@dataclass(frozen=True)
class OrderPlanEventCorruption:
    code: str
    reason: str
    line_number: int | None = None


@dataclass(frozen=True)
class OrderPlanEventVerification:
    healthy: bool
    tail_truncated: bool
    corruptions: tuple[OrderPlanEventCorruption, ...]
    event_count: int
    order_plan_ids: frozenset[str]
    events: tuple[OrderPlanEvent, ...]


@dataclass(frozen=True)
class BoundOrderPlanEventVerification:
    stream: OrderPlanEventVerification
    plan_verified: bool
    corruptions: tuple[OrderPlanEventCorruption, ...]

    @property
    def healthy(self) -> bool:
        return self.plan_verified and not self.corruptions


def verify_order_plan_event_stream(
    path: Path,
    *,
    expected_order_plan_id: str | None = None,
) -> OrderPlanEventVerification:
    """Decode and validate one immutable snapshot of an order-plan event stream."""
    raw = path.read_bytes() if path.exists() else b""
    tail_truncated = bool(raw and not raw.endswith(b"\n"))
    corruptions: list[OrderPlanEventCorruption] = []
    if tail_truncated:
        corruptions.append(
            OrderPlanEventCorruption("TRUNCATED_TAIL", "event stream lacks final newline")
        )
    events: list[OrderPlanEvent] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            events.append(OrderPlanEvent.model_validate(json.loads(line)))
        except Exception as exc:
            corruptions.append(
                OrderPlanEventCorruption(
                    "INVALID_RECORD", f"{type(exc).__name__}: {exc}", line_number
                )
            )
    order_plan_ids = frozenset(event.order_plan_id for event in events)
    if len(order_plan_ids) > 1:
        corruptions.append(
            OrderPlanEventCorruption("MIXED_ORDER_PLAN_ID", "stream contains multiple ids")
        )
    detected_id = next(iter(order_plan_ids), None)
    if expected_order_plan_id is not None and detected_id not in {None, expected_order_plan_id}:
        corruptions.append(
            OrderPlanEventCorruption("ORDER_PLAN_ID_MISMATCH", "unexpected order_plan_id")
        )
    if detected_id is not None:
        expected_stem = hashlib.sha256(detected_id.encode("utf-8")).hexdigest()
        if path.stem != expected_stem:
            corruptions.append(
                OrderPlanEventCorruption(
                    "FILENAME_ID_MISMATCH", "filename does not bind order_plan_id"
                )
            )
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        corruptions.append(
            OrderPlanEventCorruption("DUPLICATE_EVENT_ID", "event_id is not unique")
        )
    return OrderPlanEventVerification(
        healthy=not corruptions,
        tail_truncated=tail_truncated,
        corruptions=tuple(corruptions),
        event_count=len(events),
        order_plan_ids=order_plan_ids,
        events=tuple(events),
    )


def verify_bound_order_plan_event_stream_assume_locked(
    *,
    store: ArtifactStore,
    path: Path,
    expected_order_plan_id: str | None = None,
) -> BoundOrderPlanEventVerification:
    """Verify an event stream and its bound order-plan artifact under the store lock."""
    stream = verify_order_plan_event_stream(
        path,
        expected_order_plan_id=expected_order_plan_id,
    )
    corruptions = list(stream.corruptions)
    detected_id = next(iter(stream.order_plan_ids), None)

    if detected_id is None:
        if path.exists() and path.stat().st_size > 0:
            corruptions.append(
                OrderPlanEventCorruption(
                    "ORPHAN_EVENT_STREAM",
                    "event stream has no valid order_plan_id",
                )
            )
        return BoundOrderPlanEventVerification(
            stream=stream,
            plan_verified=False,
            corruptions=tuple(corruptions),
        )

    try:
        verification = store._verify_assume_locked(
            detected_id,
            expected_relative_path=f"{detected_id}.json",
        )
    except StorageValidationError:
        corruptions.append(
            OrderPlanEventCorruption(
                "MISSING_ORDER_PLAN",
                "event stream references a missing or invalid order plan manifest",
            )
        )
        return BoundOrderPlanEventVerification(
            stream=stream,
            plan_verified=False,
            corruptions=tuple(corruptions),
        )

    if verification.code == "MISSING_ARTIFACT":
        corruptions.append(
            OrderPlanEventCorruption(
                "MISSING_ORDER_PLAN",
                "event stream references a missing order plan",
            )
        )
        plan_verified = False
    elif not verification.verified:
        corruptions.append(
            OrderPlanEventCorruption(
                "INVALID_ORDER_PLAN",
                "event stream references an invalid order plan",
            )
        )
        plan_verified = False
    else:
        plan_verified = True

    return BoundOrderPlanEventVerification(
        stream=stream,
        plan_verified=plan_verified,
        corruptions=tuple(corruptions),
    )


def save_order_plan(
    plan: OrderPlan,
    *,
    artifact_store: ArtifactStore,
) -> Path:
    store = artifact_store
    content = plan.model_dump_json(indent=2).encode("utf-8")
    receipt = store.create(
        f"{plan.order_plan_id}.json",
        content,
        metadata=ArtifactMetadata(
            artifact_id=plan.order_plan_id,
            artifact_type="order_plan",
            producer="services.order_plan_service.save_order_plan",
            related_strategy_id=plan.strategy_id,
        ),
    )
    return receipt.path


def load_order_plan(
    identifier: str,
    *,
    artifact_store: ArtifactStore,
) -> OrderPlan:
    path = Path(identifier)
    store = artifact_store
    if path.is_absolute() or path.parent != Path("."):
        selected = path.expanduser().resolve()
        if selected.parent != store.root:
            raise StorageValidationError(
                store_name="order_plans",
                path=selected,
                operation="load",
                reason="order plan path is outside the artifact store root",
            )
        order_plan_id = path.stem
    else:
        order_plan_id = identifier
    relative_path = f"{order_plan_id}.json"
    raw = store.read_verified(order_plan_id, expected_relative_path=relative_path)
    try:
        plan = OrderPlan.model_validate_json(raw)
    except Exception:
        raise
    if plan.order_plan_id != order_plan_id:
        raise ValueError("order plan id does not match repository path")
    return plan


def append_order_plan_event(
    order_plan_id: str,
    *,
    event_type: str,
    actor: str,
    details: dict[str, object] | None = None,
    artifact_store: ArtifactStore,
) -> OrderPlanEvent:
    store = artifact_store
    with store.lock_manager.resource_lock(store._resource):
        verification = store._verify_assume_locked(
            order_plan_id,
            expected_relative_path=f"{order_plan_id}.json",
        )
        if not verification.verified:
            store._read_verified_assume_locked(
                order_plan_id, expected_relative_path=f"{order_plan_id}.json"
            )
        event_path = _event_path(store, order_plan_id)
        if event_path.exists():
            stream = verify_order_plan_event_stream(
                event_path, expected_order_plan_id=order_plan_id
            )
            if not stream.healthy:
                raise StorageCorruptError(
                    store_name="order_plan_events",
                    path=event_path,
                    operation="append",
                    reason="cannot append to corrupt event stream",
                )
        event = OrderPlanEvent(
            order_plan_id=order_plan_id,
            event_type=event_type,
            actor=actor,
            details=details or {},
        )
        store.atomic_store.append_jsonl_assume_locked(
            event_path, event.model_dump(mode="json")
        )
        return event


def load_order_plan_events(
    order_plan_id: str,
    *,
    artifact_store: ArtifactStore,
) -> list[OrderPlanEvent]:
    store = artifact_store
    path = _event_path(store, order_plan_id)
    with store.lock_manager.resource_lock(store._resource):
        if not path.exists():
            return []
        verification = verify_bound_order_plan_event_stream_assume_locked(
            store=store,
            path=path,
            expected_order_plan_id=order_plan_id,
        )
        if verification.stream.tail_truncated:
            raise StorageCorruptError(
                store_name="order_plan_events",
                path=path,
                operation="read",
                reason="order plan event stream has a truncated tail",
                suggested_repair="inspect and restore the event stream before execution",
            )
        if not verification.healthy:
            reason = "; ".join(item.code for item in verification.corruptions)
            raise StorageCorruptError(
                store_name="order_plan_events",
                path=path,
                operation="read",
                reason=f"order plan event stream is corrupt or unbound: {reason}",
            )
        return list(verification.stream.events)


def _event_path(store: ArtifactStore, order_plan_id: str) -> Path:
    digest = hashlib.sha256(order_plan_id.encode("utf-8")).hexdigest()
    return store.path_for(f".events/{digest}.jsonl")
