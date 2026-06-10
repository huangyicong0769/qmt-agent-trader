"""Strategy approval state machine and approval files."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus

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


def write_approval_file(approval: StrategyApproval, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{approval.strategy_id}_{approval.strategy_version}.approval.yaml"
    path.write_text(
        yaml.safe_dump(approval.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def read_approval_file(path: Path) -> StrategyApproval:
    return StrategyApproval.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
