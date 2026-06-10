"""Immutable order plan model."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.core.ids import new_id, new_idempotency_key, shanghai_now_iso
from qmt_agent_trader.core.security import canonical_json
from qmt_agent_trader.core.types import ApprovalStatus, RiskStatus


class RiskCheckResult(BaseModel):
    model_config = {"frozen": True}

    name: str
    status: RiskStatus
    message: str | None = None


class RiskChecks(BaseModel):
    model_config = {"frozen": True}

    status: RiskStatus
    checks: tuple[RiskCheckResult, ...] = Field(default_factory=tuple)

    @classmethod
    def passed(cls, names: list[str]) -> RiskChecks:
        return cls(
            status=RiskStatus.PASSED,
            checks=tuple(RiskCheckResult(name=name, status=RiskStatus.PASSED) for name in names),
        )


class OrderPlanApproval(BaseModel):
    model_config = {"frozen": True}

    required: bool = True
    status: ApprovalStatus = ApprovalStatus.DRAFT
    approved_by: str | None = None
    approved_at: str | None = None


class OrderPlan(BaseModel):
    model_config = {"frozen": True}

    order_plan_id: str = Field(default_factory=lambda: new_id("op"))
    created_at: str = Field(default_factory=shanghai_now_iso)
    strategy_id: str
    strategy_version: str
    strategy_approval_status: ApprovalStatus
    account_id_hash: str
    dry_run: bool = True
    orders: tuple[Order, ...]
    risk_checks: RiskChecks
    approval: OrderPlanApproval
    idempotency_key: str = Field(default_factory=new_idempotency_key)
    plan_hash: str | None = None

    def model_post_init(self, __context: object) -> None:
        expected_hash = self.compute_hash(include_existing_hash=False)
        if self.plan_hash is not None and self.plan_hash != expected_hash:
            raise ValueError("order plan hash mismatch")
        object.__setattr__(self, "plan_hash", expected_hash)

    def canonical_payload(self, include_existing_hash: bool = True) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_existing_hash:
            payload.pop("plan_hash", None)
        return payload

    def compute_hash(self, include_existing_hash: bool = False) -> str:
        payload = self.canonical_payload(include_existing_hash=include_existing_hash)
        return sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def assert_submittable(self, live: bool = False) -> None:
        if self.strategy_approval_status != ApprovalStatus.APPROVED:
            raise ValueError("strategy is not approved")
        if self.risk_checks.status != RiskStatus.PASSED:
            raise ValueError("risk checks did not pass")
        if self.approval.status != ApprovalStatus.APPROVED:
            raise ValueError("order plan is not approved")
        if live and self.dry_run:
            raise ValueError("dry-run plan cannot be submitted as live")
