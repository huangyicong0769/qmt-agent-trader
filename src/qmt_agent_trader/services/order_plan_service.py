"""Order plan generation service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.broker.order_plan import OrderPlan, OrderPlanApproval, RiskChecks
from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus, OrderType, Side
from qmt_agent_trader.persistence.artifacts import (
    ArtifactMetadata,
    ArtifactStore,
    artifact_store_for_root,
)


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
    order_plan_id: str
    event_type: str
    actor: str
    created_at: str = Field(default_factory=shanghai_now_iso)
    details: dict[str, object] = Field(default_factory=dict)


def save_order_plan(
    plan: OrderPlan,
    directory: Path,
    *,
    artifact_store: ArtifactStore | None = None,
) -> Path:
    store = artifact_store or artifact_store_for_root(directory)
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
    directory: Path = Path("order_plans"),
    *,
    artifact_store: ArtifactStore | None = None,
) -> OrderPlan:
    path = Path(identifier)
    if path.exists():
        selected_directory = path.parent
        order_plan_id = path.stem
    else:
        selected_directory = directory
        order_plan_id = identifier
    store = artifact_store or artifact_store_for_root(selected_directory)
    try:
        content = store.read_verified(order_plan_id)
    except Exception as exc:
        if not store.path_for(f"{order_plan_id}.json").exists():
            raise ValueError(f"order plan not found: {identifier}") from exc
        raise
    return OrderPlan.model_validate_json(content)


def append_order_plan_event(
    order_plan_id: str,
    *,
    directory: Path = Path("order_plans"),
    event_type: str,
    actor: str,
    details: dict[str, object] | None = None,
    artifact_store: ArtifactStore | None = None,
) -> OrderPlanEvent:
    store = artifact_store or artifact_store_for_root(directory)
    verification = store.verify(order_plan_id)
    if not verification.verified:
        store.read_verified(order_plan_id)
    event = OrderPlanEvent(
        order_plan_id=order_plan_id,
        event_type=event_type,
        actor=actor,
        details=details or {},
    )
    event_path = _event_path(store, order_plan_id)
    store.atomic_store.append_jsonl(event_path, event.model_dump(mode="json"))
    return event


def load_order_plan_events(
    order_plan_id: str,
    directory: Path = Path("order_plans"),
    *,
    artifact_store: ArtifactStore | None = None,
) -> list[OrderPlanEvent]:
    store = artifact_store or artifact_store_for_root(directory)
    path = _event_path(store, order_plan_id)
    if not path.exists():
        return []
    events: list[OrderPlanEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        events.append(OrderPlanEvent.model_validate(json.loads(line)))
    return events


def _event_path(store: ArtifactStore, order_plan_id: str) -> Path:
    digest = hashlib.sha256(order_plan_id.encode("utf-8")).hexdigest()
    return store.path_for(f".events/{digest}.jsonl")
