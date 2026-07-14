"""Point-in-time rolling universe membership."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError


def _parse_date_key(value: str) -> date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"invalid rolling universe date key: {value}")


@dataclass(frozen=True)
class RollingUniverseTimeline:
    dates: tuple[date, ...]
    membership_by_date: dict[date, tuple[str, ...]]

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Sequence[str]],
    ) -> RollingUniverseTimeline:
        normalized = {
            _parse_date_key(str(key)): tuple(dict.fromkeys(str(item) for item in symbols))
            for key, symbols in mapping.items()
        }
        return cls(
            dates=tuple(sorted(normalized)),
            membership_by_date=normalized,
        )

    def membership_as_of(self, as_of_date: date) -> tuple[str, ...]:
        index = bisect_right(self.dates, as_of_date) - 1
        if index < 0:
            raise BacktestUniverseIntegrityError(
                code="ROLLING_UNIVERSE_SNAPSHOT_NOT_AVAILABLE",
                message="no rolling-universe snapshot exists on or before signal date",
                trade_date=f"{as_of_date:%Y-%m-%d}",
                field="symbols_by_date",
                details={
                    "first_available_snapshot": (
                        f"{self.dates[0]:%Y-%m-%d}" if self.dates else None
                    )
                },
            )
        return self.membership_by_date[self.dates[index]]
