"""Persistent strategy registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.repositories.versioned_json import (
    RegistrySnapshot,
    VersionedJsonRegistry,
)
from qmt_agent_trader.strategy.approval import transition_status
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategyKind,
    StrategySource,
    StrategySpec,
)


class StrategyRegistry:
    """Registry for built-in, human-authored, and agent-generated strategies."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        lock_manager: LockManager | None = None,
        atomic_store: AtomicFileStore | None = None,
    ) -> None:
        self.root = root or get_settings().resolved_data_dir / "strategies"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        manager = lock_manager or LockManager(self.root / ".locks")
        store = atomic_store or AtomicFileStore(manager)
        self._repository = VersionedJsonRegistry(
            path=self.registry_path,
            item_loader=_load_file_strategy,
            item_dumper=lambda item: item.model_dump(mode="json"),
            item_identity=lambda item: item.strategy_id,
            legacy_items_key="strategies",
            lock_manager=manager,
            atomic_store=store,
            store_name="strategy_registry",
        )
        self._runtime_strategies: dict[str, Any] = {}
        self._saved: dict[str, SavedStrategy] = _builtin_strategies()
        self._apply_snapshot(self._repository.load_snapshot())

    def register(self, strategy_id: str, strategy: Any) -> None:
        """Backward-compatible in-memory registration."""
        self._refresh()
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
        self._refresh()
        values = list(self._saved.values())
        if not include_builtins:
            values = [item for item in values if item.source != StrategySource.BUILTIN]
        return sorted(values, key=lambda item: (item.strategy_id, item.version))

    def get_strategy(self, strategy_id: str) -> SavedStrategy | None:
        self._refresh()
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
        if saved.source == StrategySource.BUILTIN:
            raise ValueError("built-in strategies are managed by code")
        if saved.status == ApprovalStatus.APPROVED:
            raise ValueError("agent-generated strategies cannot be saved as APPROVED")
        now = shanghai_now_iso()
        stored = saved.model_copy(update={"created_at": saved.created_at or now, "updated_at": now})

        def save(items: list[SavedStrategy]) -> list[SavedStrategy]:
            if stored.strategy_id in _builtin_strategies() or any(
                item.strategy_id == stored.strategy_id for item in items
            ):
                raise ValueError(f"strategy already registered: {stored.strategy_id}")
            return [*items, stored]

        self._apply_snapshot(self._repository.mutate(save))
        return stored

    def update_status(
        self,
        strategy_id: str,
        status: ApprovalStatus,
        *,
        trusted: bool = False,
    ) -> SavedStrategy:
        if status == ApprovalStatus.APPROVED and not trusted:
            raise ValueError("APPROVED requires trusted approval workflow")

        def update(saved: SavedStrategy) -> SavedStrategy:
            target = transition_status(saved.status, status)
            return saved.model_copy(update={"status": target, "updated_at": shanghai_now_iso()})

        return self._update_record(strategy_id, update)

    def attach_report(self, strategy_id: str, report_path: str) -> SavedStrategy:
        def update(saved: SavedStrategy) -> SavedStrategy:
            report_paths = [*saved.report_paths]
            if report_path not in report_paths:
                report_paths.append(report_path)
            return saved.model_copy(
                update={"report_paths": report_paths, "updated_at": shanghai_now_iso()}
            )

        return self._update_record(strategy_id, update)

    def attach_generated_implementation(
        self,
        strategy_id: str,
        *,
        spec: StrategySpec,
        code_path: str,
        tests_path: str | None,
    ) -> SavedStrategy:
        def update(saved: SavedStrategy) -> SavedStrategy:
            if saved.source != StrategySource.AGENT_GENERATED:
                raise ValueError("generated implementations can only update agent-generated drafts")
            if saved.status == ApprovalStatus.APPROVED:
                raise ValueError("APPROVED strategies require trusted approval workflow")
            return saved.model_copy(
                update={
                    "name": spec.name,
                    "version": spec.version,
                    "spec": spec,
                    "implementation_ref": f"file:{code_path}",
                    "code_path": code_path,
                    "tests_path": tests_path,
                    "updated_at": shanghai_now_iso(),
                }
            )

        return self._update_record(strategy_id, update)

    def attach_approval(self, strategy_id: str, approval_file: str) -> SavedStrategy:
        return self._update_record(
            strategy_id,
            lambda saved: saved.model_copy(
                update={"approval_file": approval_file, "updated_at": shanghai_now_iso()}
            ),
        )

    def _require_mutable(
        self,
        strategy_id: str,
        current: list[SavedStrategy],
    ) -> SavedStrategy:
        saved = next((item for item in current if item.strategy_id == strategy_id), None)
        if saved is None:
            saved = _builtin_strategies().get(strategy_id)
        if saved is None:
            raise ValueError(f"strategy not found: {strategy_id}")
        if saved.source == StrategySource.BUILTIN:
            raise ValueError("built-in strategies cannot be modified in registry")
        return saved

    def _update_record(
        self,
        strategy_id: str,
        update: Callable[[SavedStrategy], SavedStrategy],
    ) -> SavedStrategy:
        result: SavedStrategy | None = None

        def mutate(items: list[SavedStrategy]) -> list[SavedStrategy]:
            nonlocal result
            saved = self._require_mutable(strategy_id, items)
            result = update(saved)
            return [result if item.strategy_id == strategy_id else item for item in items]

        self._apply_snapshot(self._repository.mutate(mutate))
        assert result is not None
        return result

    def _refresh(self) -> None:
        self._apply_snapshot(self._repository.load_snapshot())

    def _apply_snapshot(self, snapshot: RegistrySnapshot[SavedStrategy]) -> None:
        saved = _builtin_strategies()
        for item in snapshot.items:
            saved[item.strategy_id] = item
        self._saved = saved


def _load_file_strategy(data: dict[str, Any]) -> SavedStrategy:
    saved = SavedStrategy.model_validate(data)
    if saved.strategy_id in _builtin_strategies() or saved.source == StrategySource.BUILTIN:
        raise ValueError("built-in strategies are managed by code")
    return saved


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
