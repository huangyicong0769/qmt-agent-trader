"""Factor validation cache — avoids redundant backtests.

Stores validation results keyed by (factor_name, start_date, end_date).
When the LLM repeats identical backtest requests, cached results are returned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_ROOT = Path("reports/cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _cache_key(factor_name: str, start: str, end: str) -> str:
    raw = f"{factor_name}|{start}|{end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached_validation(
    factor_name: str, start: str, end: str
) -> dict[str, Any] | None:
    """Return cached validation result, or None if not found."""
    key = _cache_key(factor_name, start, end)
    cache_path = CACHE_ROOT / f"factor_{key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("result"), dict):
            return dict(data["result"])
        return dict(data)
    except Exception:
        cache_path.unlink(missing_ok=True)
        return None


def put_cached_validation(
    factor_name: str, start: str, end: str, result: dict[str, Any]
) -> None:
    """Store validation result in cache."""
    key = _cache_key(factor_name, start, end)
    cache_path = CACHE_ROOT / f"factor_{key}.json"
    cache_path.write_text(
        json.dumps(
            {
                "factor_name": factor_name,
                "start_date": start,
                "end_date": end,
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
