"""Dataset catalog metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

LEGACY_RAW_BATCH_PATTERNS = (
    re.compile(r"^tushare_daily_\d{8}_\d{8}$"),
    re.compile(r"^tushare_suspend_\d{8}_\d{8}$"),
    re.compile(r"^tushare_stk_limit_\d{8}_\d{8}$"),
)


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    layer: str
    path: Path
    version: str
    source: str


def visible_dataset_names(layer: str, names: list[str]) -> list[str]:
    if layer != "raw":
        return names
    return [name for name in names if not is_legacy_raw_batch_name(name)]


def is_legacy_raw_batch_name(name: str) -> bool:
    return any(pattern.match(name) for pattern in LEGACY_RAW_BATCH_PATTERNS)
