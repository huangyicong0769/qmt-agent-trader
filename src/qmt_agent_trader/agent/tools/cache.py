"""Factor validation cache — avoids redundant backtests.

Stores validation results keyed by (factor_name, start_date, end_date).
When the LLM repeats identical backtest requests, cached results are returned.
"""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.persistence.cache import ContentAddressedCache

_REQUIRED_VALIDATION_FIELDS = {
    "actual_data_start",
    "actual_data_end",
    "data_freshness",
}


def _cache_key(
    cache: ContentAddressedCache, factor_name: str, start: str, end: str
) -> str:
    return cache.key_for({"factor_name": factor_name, "start": start, "end": end})


def get_cached_validation(
    factor_name: str, start: str, end: str, cache: ContentAddressedCache
) -> dict[str, Any] | None:
    """Return cached validation result, or None if not found."""
    key = _cache_key(cache, factor_name, start, end)
    result = cache.get("factor-validation", key)
    if result is None:
        return None
    if result.get("status") == "validated" and not _REQUIRED_VALIDATION_FIELDS.issubset(result):
        cache.invalidate(
            "factor-validation",
            key,
            expected_value=result,
            reason="CACHE_VALIDATION_STALE",
        )
        return None
    return result


def put_cached_validation(
    factor_name: str,
    start: str,
    end: str,
    result: dict[str, Any],
    cache: ContentAddressedCache,
) -> None:
    """Store validation result in cache."""
    key = _cache_key(cache, factor_name, start, end)
    cache.put("factor-validation", key, result)
