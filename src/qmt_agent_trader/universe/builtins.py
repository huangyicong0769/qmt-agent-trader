"""Built-in broad universe specs.

This module intentionally contains no thematic baskets.
"""

from __future__ import annotations

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.universe.models import UniverseSelection, UniverseSpec


def broad_universe_spec(
    universe_type: str,
    *,
    mode: str = "snapshot",
    rebalance_frequency: str = "daily",
) -> UniverseSpec:
    normalized = universe_type.lower()
    if normalized in {"all", "stock_etf", "mixed"}:
        asset_types = ["stock", "etf"]
        universe_id = "builtin_mixed_all"
        name = "All stocks and ETFs"
    elif normalized == "stock":
        asset_types = ["stock"]
        universe_id = "builtin_stock_all"
        name = "All stocks"
    elif normalized == "etf":
        asset_types = ["etf"]
        universe_id = "builtin_etf_all"
        name = "All ETFs"
    else:
        raise ValueError(f"unsupported broad universe type: {universe_type}")
    return UniverseSpec(
        universe_id=universe_id,
        name=name,
        description="Broad built-in universe resolved from available data lake bars.",
        source="builtin",
        asset_types=asset_types,
        selection=UniverseSelection(mode="all"),
        mode=mode,
        rebalance_frequency=rebalance_frequency,
        created_by="system",
        created_at=shanghai_now_iso(),
    )
