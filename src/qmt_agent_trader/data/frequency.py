"""Frequency helpers for target-frequency data alignment."""

from __future__ import annotations

from enum import StrEnum


class Frequency(StrEnum):
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    EVENT = "event"
    UNKNOWN = "unknown"


_FREQUENCY_RANK = {
    Frequency.ANNUAL: 1,
    Frequency.QUARTERLY: 2,
    Frequency.MONTHLY: 3,
    Frequency.WEEKLY: 4,
    Frequency.DAILY: 5,
    Frequency.INTRADAY: 6,
}


def is_lower_frequency(source: Frequency, target: Frequency) -> bool:
    """Return whether ``source`` should be as-of aligned to ``target``."""

    if source in {Frequency.EVENT, Frequency.UNKNOWN}:
        return False
    if target in {Frequency.EVENT, Frequency.UNKNOWN}:
        return False
    return _FREQUENCY_RANK[source] < _FREQUENCY_RANK[target]
