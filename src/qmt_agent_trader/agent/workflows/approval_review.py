"""Approval review workflow skeleton."""

from __future__ import annotations


def summarize_for_review(strategy_id: str) -> dict[str, str]:
    return {"strategy_id": strategy_id, "recommendation": "human_review_required"}
