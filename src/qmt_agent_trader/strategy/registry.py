"""Strategy registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyRegistry:
    strategies: dict[str, Any] = field(default_factory=dict)

    def register(self, strategy_id: str, strategy: Any) -> None:
        if strategy_id in self.strategies:
            raise ValueError(f"strategy already registered: {strategy_id}")
        self.strategies[strategy_id] = strategy

    def get(self, strategy_id: str) -> Any:
        return self.strategies[strategy_id]
