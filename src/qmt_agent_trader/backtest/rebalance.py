"""Trading-calendar scheduling for research rebalances."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

RebalanceFrequency = Literal["daily", "weekly", "monthly"]


def select_signal_dates(
    dates: Sequence[date],
    frequency: RebalanceFrequency,
) -> tuple[date, ...]:
    ordered = tuple(sorted(dict.fromkeys(dates)))
    if frequency == "daily":
        return ordered
    last_by_bucket: dict[tuple[int, int], date] = {}
    for item in ordered:
        if frequency == "weekly":
            iso = item.isocalendar()
            bucket = (iso.year, iso.week)
        elif frequency == "monthly":
            bucket = (item.year, item.month)
        else:
            raise ValueError(f"unsupported rebalance frequency: {frequency}")
        last_by_bucket[bucket] = item
    return tuple(sorted(last_by_bucket.values()))


def build_execution_schedule(
    dates: Sequence[date],
    *,
    signal_dates: Sequence[date],
    delay_days: int,
) -> dict[date, date]:
    if delay_days < 1:
        raise ValueError("execution delay must be at least one trading session")
    ordered = tuple(sorted(dict.fromkeys(dates)))
    index_by_date = {item: index for index, item in enumerate(ordered)}
    schedule: dict[date, date] = {}
    for signal_date in signal_dates:
        execution_index = index_by_date[signal_date] + delay_days
        if execution_index < len(ordered):
            schedule[ordered[execution_index]] = signal_date
    return schedule
