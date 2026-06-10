"""Shared type aliases and enums."""

from __future__ import annotations

from enum import StrEnum
from typing import NewType

Symbol = NewType("Symbol", str)
StrategyId = NewType("StrategyId", str)
OrderPlanId = NewType("OrderPlanId", str)


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ApprovalStatus(StrEnum):
    DRAFT = "DRAFT"
    GENERATED_BY_LLM = "GENERATED_BY_LLM"
    BACKTESTED = "BACKTESTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class RiskStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class ExecutionMode(StrEnum):
    DRY_RUN = "DRY_RUN"
    PAPER = "PAPER"
    LIVE = "LIVE"
