"""Strategy approval state machine and approval files."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.persistence.artifacts import (
    ArtifactMetadata,
    ArtifactStore,
)
from qmt_agent_trader.persistence.errors import StorageConflictError, StorageValidationError

ALLOWED_TRANSITIONS: dict[ApprovalStatus, set[ApprovalStatus]] = {
    ApprovalStatus.DRAFT: {
        ApprovalStatus.GENERATED_BY_LLM,
        ApprovalStatus.BACKTESTED,
        ApprovalStatus.REJECTED,
    },
    ApprovalStatus.GENERATED_BY_LLM: {ApprovalStatus.BACKTESTED, ApprovalStatus.REJECTED},
    ApprovalStatus.BACKTESTED: {ApprovalStatus.REVIEW_REQUIRED, ApprovalStatus.REJECTED},
    ApprovalStatus.REVIEW_REQUIRED: {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED},
    ApprovalStatus.APPROVED: {ApprovalStatus.RETIRED},
    ApprovalStatus.REJECTED: set(),
    ApprovalStatus.RETIRED: set(),
}


class StrategyApproval(BaseModel):
    strategy_id: str
    strategy_name: str
    strategy_version: str
    approved_by: str
    approved_at: str = Field(default_factory=shanghai_now_iso)
    allowed_universe: list[str]
    allowed_accounts: list[str]
    max_single_position_pct: float = Field(ge=0, le=1)
    max_turnover_daily_pct: float = Field(ge=0, le=1)
    max_order_value: float = Field(gt=0)
    live_trading_allowed: bool = False
    paper_trading_allowed: bool = True
    notes: str = ""


def transition_status(current: ApprovalStatus, target: ApprovalStatus) -> ApprovalStatus:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"invalid strategy status transition: {current} -> {target}")
    return target


def write_approval_file(
    approval: StrategyApproval,
    directory: Path,
    *,
    artifact_store: ArtifactStore,
) -> Path:
    filename = f"{approval.strategy_id}_{approval.strategy_version}.approval.yaml"
    store = artifact_store
    content = yaml.safe_dump(
        approval.model_dump(mode="json"), sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    if store.path_for(filename).exists():
        existing = read_approval_file(store.path_for(filename), artifact_store=store)
        requested = approval.model_dump(mode="json", exclude={"approved_at"})
        persisted = existing.model_dump(mode="json", exclude={"approved_at"})
        if requested != persisted:
            raise StorageConflictError(
                store_name="approvals",
                path=store.path_for(filename),
                operation="resume",
                reason="existing immutable approval differs from requested approval",
            )
        return store.path_for(filename)
    receipt = store.create(
        filename,
        content,
        metadata=ArtifactMetadata(
            artifact_id=_approval_artifact_id(filename),
            artifact_type="strategy_approval",
            producer="strategy.approval.write_approval_file",
            related_strategy_id=approval.strategy_id,
        ),
    )
    return receipt.path


def read_approval_file(
    path: Path,
    *,
    artifact_store: ArtifactStore,
) -> StrategyApproval:
    store = artifact_store
    artifact_id = _approval_artifact_id(path.name)
    raw = store.read_verified(artifact_id, expected_relative_path=path.name)
    try:
        approval = StrategyApproval.model_validate(yaml.safe_load(raw))
    except Exception as exc:
        raise StorageValidationError(
            store_name="approvals",
            path=path,
            operation="read",
            reason="approval is invalid",
            original_error=exc,
        ) from exc
    if _approval_filename(approval) != path.name:
        raise StorageValidationError(
            store_name="approvals",
            path=path,
            operation="read",
            reason="approval filename identity does not match parsed strategy",
        )
    return approval


def _approval_artifact_id(filename: str) -> str:
    return f"approval:{filename}"


def _approval_filename(approval: StrategyApproval) -> str:
    return f"{approval.strategy_id}_{approval.strategy_version}.approval.yaml"
