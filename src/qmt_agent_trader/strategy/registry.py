"""Persistent strategy registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.strategy.approval import transition_status
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategyKind,
    StrategySource,
    StrategySpec,
)


class StrategyRegistry:
    """Registry for built-in, human-authored, and agent-generated strategies."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().resolved_data_dir / "strategies"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self._runtime_strategies: dict[str, Any] = {}
        self._saved: dict[str, SavedStrategy] = _builtin_strategies()
        self._saved.update(self._load_file_registry())

    def register(self, strategy_id: str, strategy: Any) -> None:
        """Backward-compatible in-memory registration."""
        if strategy_id in self._runtime_strategies or strategy_id in self._saved:
            raise ValueError(f"strategy already registered: {strategy_id}")
        self._runtime_strategies[strategy_id] = strategy

    def get(self, strategy_id: str) -> Any:
        if strategy_id in self._runtime_strategies:
            return self._runtime_strategies[strategy_id]
        saved = self.get_strategy(strategy_id)
        if saved is None:
            raise KeyError(strategy_id)
        return saved

    def list_strategies(self, include_builtins: bool = True) -> list[SavedStrategy]:
        values = list(self._saved.values())
        if not include_builtins:
            values = [item for item in values if item.source != StrategySource.BUILTIN]
        return sorted(values, key=lambda item: (item.strategy_id, item.version))

    def get_strategy(self, strategy_id: str) -> SavedStrategy | None:
        return self._saved.get(strategy_id)

    def find_strategies(self, query: str | None = None) -> list[SavedStrategy]:
        needle = str(query or "").strip().lower()
        strategies = self.list_strategies()
        if not needle:
            return strategies
        return [
            item
            for item in strategies
            if needle in item.strategy_id.lower()
            or needle in item.name.lower()
            or needle in item.spec.description.lower()
        ]

    def save_candidate(self, saved: SavedStrategy) -> SavedStrategy:
        if saved.strategy_id in self._saved:
            raise ValueError(f"strategy already registered: {saved.strategy_id}")
        if saved.source == StrategySource.BUILTIN:
            raise ValueError("built-in strategies are managed by code")
        if saved.status == ApprovalStatus.APPROVED:
            raise ValueError("agent-generated strategies cannot be saved as APPROVED")
        now = shanghai_now_iso()
        stored = saved.model_copy(update={"created_at": saved.created_at or now, "updated_at": now})
        self._saved[stored.strategy_id] = stored
        self._persist_file_registry()
        return stored

    def update_status(
        self,
        strategy_id: str,
        status: ApprovalStatus,
        *,
        trusted: bool = False,
    ) -> SavedStrategy:
        saved = self._require_mutable(strategy_id)
        if status == ApprovalStatus.APPROVED and not trusted:
            raise ValueError("APPROVED requires trusted approval workflow")
        target = transition_status(saved.status, status)
        updated = saved.model_copy(update={"status": target, "updated_at": shanghai_now_iso()})
        self._saved[strategy_id] = updated
        self._persist_file_registry()
        return updated

    def attach_report(self, strategy_id: str, report_path: str) -> SavedStrategy:
        saved = self._require_mutable(strategy_id)
        report_paths = [*saved.report_paths]
        if report_path not in report_paths:
            report_paths.append(report_path)
        updated = saved.model_copy(
            update={"report_paths": report_paths, "updated_at": shanghai_now_iso()}
        )
        self._saved[strategy_id] = updated
        self._persist_file_registry()
        return updated

    def attach_approval(self, strategy_id: str, approval_file: str) -> SavedStrategy:
        saved = self._require_mutable(strategy_id)
        updated = saved.model_copy(
            update={"approval_file": approval_file, "updated_at": shanghai_now_iso()}
        )
        self._saved[strategy_id] = updated
        self._persist_file_registry()
        return updated

    def _require_mutable(self, strategy_id: str) -> SavedStrategy:
        saved = self.get_strategy(strategy_id)
        if saved is None:
            raise ValueError(f"strategy not found: {strategy_id}")
        if saved.source == StrategySource.BUILTIN:
            raise ValueError("built-in strategies cannot be modified in registry")
        return saved

    def _load_file_registry(self) -> dict[str, SavedStrategy]:
        if not self.registry_path.exists():
            return {}
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        records = payload.get("strategies", [])
        if not isinstance(records, list):
            return {}
        loaded: dict[str, SavedStrategy] = {}
        for item in records:
            if isinstance(item, dict):
                saved = SavedStrategy.model_validate(item)
                loaded[saved.strategy_id] = saved
        return loaded

    def _persist_file_registry(self) -> None:
        strategies = [
            item.model_dump(mode="json")
            for item in self.list_strategies(include_builtins=False)
        ]
        payload = {"version": 1, "strategies": strategies}
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def _builtin_strategies() -> dict[str, SavedStrategy]:
    records = [
        _builtin_record(
            strategy_id="factor_rank_long_only_v1",
            name="Factor Rank Long Only",
            kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
            implementation_ref="builtin:qmt_agent_trader.strategy.examples.factor_rank_long_only",
        ),
        _builtin_record(
            strategy_id="etf_trend_v1",
            name="ETF Trend",
            kind=StrategyKind.ETF_TREND,
            implementation_ref="builtin:qmt_agent_trader.strategy.examples.etf_trend",
        ),
    ]
    return {item.strategy_id: item for item in records}


def _builtin_record(
    *,
    strategy_id: str,
    name: str,
    kind: StrategyKind,
    implementation_ref: str,
) -> SavedStrategy:
    spec = StrategySpec(
        strategy_id=strategy_id,
        name=name,
        version="1.0.0",
        kind=kind,
        source=StrategySource.BUILTIN,
    )
    return SavedStrategy(
        strategy_id=strategy_id,
        name=name,
        version="1.0.0",
        source=StrategySource.BUILTIN,
        status=ApprovalStatus.DRAFT,
        spec=spec,
        implementation_ref=implementation_ref,
        created_by="system",
    )
