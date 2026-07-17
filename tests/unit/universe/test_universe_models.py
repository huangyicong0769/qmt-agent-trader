from __future__ import annotations

import pytest
from pydantic import ValidationError

from qmt_agent_trader.universe.models import (
    UniverseRule,
    UniverseSelection,
    UniverseSpec,
)


def test_universe_spec_supports_snapshot_mode() -> None:
    spec = UniverseSpec(
        universe_id="u_snapshot",
        name="Snapshot stock universe",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        mode="snapshot",
        created_at="2026-07-09T00:00:00+08:00",
    )

    assert spec.mode == "snapshot"
    assert spec.research_only is True
    assert spec.live_trading_allowed is False
    assert spec.approval_required is True


def test_universe_spec_supports_rolling_mode() -> None:
    spec = UniverseSpec(
        universe_id="u_rolling",
        name="Rolling stock universe",
        source="agent_generated",
        asset_types=["stock"],
        selection=UniverseSelection(mode="all"),
        mode="rolling",
        rebalance_frequency="weekly",
        created_at="2026-07-09T00:00:00+08:00",
    )

    assert spec.mode == "rolling"
    assert spec.rebalance_frequency == "weekly"


def test_universe_rule_rejects_arbitrary_operator() -> None:
    with pytest.raises(ValidationError):
        UniverseRule(field="amount", operator="python_eval", value="__import__('os')")


def test_etf_category_requires_category_values() -> None:
    with pytest.raises(ValueError, match="etf_category selection requires categories"):
        UniverseSpec.model_validate(
            {
                "universe_id": "etf-category-empty",
                "name": "ETF category empty",
                "source": "user_defined",
                "asset_types": ["etf"],
                "selection": {
                    "mode": "etf_category",
                },
            }
        )
