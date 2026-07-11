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
    artifact_store_for_root,
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
    artifact_store: ArtifactStore | None = None,
) -> Path:
    filename = f"{approval.strategy_id}_{approval.strategy_version}.approval.yaml"
    store = artifact_store or artifact_store_for_root(directory)
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
    artifact_store: ArtifactStore | None = None,
) -> StrategyApproval:
    store = artifact_store or artifact_store_for_root(path.parent)
    artifact_id = _approval_artifact_id(path.name)
    if store.manifest_path_for(artifact_id).exists():
        raw = store.read_verified(artifact_id, expected_relative_path=path.name)
    else:
        raw = path.read_bytes()
    try:
        approval = StrategyApproval.model_validate(yaml.safe_load(raw))
    except Exception as exc:
        raise StorageValidationError(
            store_name="approvals",
            path=path,
            operation="adopt_legacy",
            reason="legacy approval is invalid",
            original_error=exc,
        ) from exc
    store.adopt(
        path.name,
        metadata=ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type="strategy_approval",
            producer="strategy.approval.legacy_adoption",
            related_strategy_id=approval.strategy_id,
        ),
    )
    content = store.read_verified(
        artifact_id,
        expected_relative_path=path.name,
    )
    return StrategyApproval.model_validate(yaml.safe_load(content))


def _approval_artifact_id(filename: str) -> str:
    return f"approval:{filename}"
