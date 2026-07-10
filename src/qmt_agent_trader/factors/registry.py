"""Saved factor registry for built-in and agent-authored research factors."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.data.contracts import (
    AlignmentPolicy,
    CoveragePolicy,
    EntityScope,
    FactorInputRequirement,
    StalenessPolicy,
    TargetCalendar,
)
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.factors.library.price_volume import (
    amount_zscore_20d,
    momentum,
    reversal_5d,
    turnover_20d,
    volatility_20d,
)
from qmt_agent_trader.factors.library.quality import (
    debt_to_assets_rank,
    gross_margin_rank,
    roe_rank,
)
from qmt_agent_trader.factors.library.value import (
    dividend_yield,
    pb_rank,
    pe_ttm_rank,
    size_log_mktcap,
)
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.repositories.versioned_json import (
    RegistrySnapshot,
    VersionedJsonRegistry,
)

FactorFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class SavedFactor:
    factor_id: str
    name: str
    version: str
    implementation_ref: str
    required_columns: tuple[str, ...]
    lookback: int
    params: dict[str, Any]
    created_by: str
    created_at: str
    status: str = "saved"
    input_requirements: tuple[FactorInputRequirement, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SavedFactor:
        return cls(
            factor_id=str(data["factor_id"]),
            name=str(data.get("name") or data["factor_id"]),
            version=str(data.get("version") or "0.1.0"),
            implementation_ref=str(data["implementation_ref"]),
            required_columns=tuple(str(item) for item in data.get("required_columns", ())),
            lookback=int(data.get("lookback") or 0),
            params=dict(data.get("params") or {}),
            created_by=str(data.get("created_by") or "unknown"),
            created_at=str(data.get("created_at") or shanghai_now_iso()),
            status=str(data.get("status") or "saved"),
            input_requirements=tuple(
                FactorInputRequirement.model_validate(item)
                for item in data.get("input_requirements", ())
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["required_columns"] = list(self.required_columns)
        data["input_requirements"] = [
            requirement.model_dump(mode="json")
            for requirement in self.input_requirements
        ]
        return data


class FactorRegistry:
    """Registry of saved factors.

    Built-in factors and agent-saved factors share the same public lookup and
    compute path. Draft files are intentionally invisible until saved here.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        lock_manager: LockManager | None = None,
        atomic_store: AtomicFileStore | None = None,
    ) -> None:
        self.root = root
        self.registry_path = root / "registry.json" if root is not None else None
        self._repository: VersionedJsonRegistry[SavedFactor] | None = None
        self._saved = _builtin_saved_factors()
        if root is not None and self.registry_path is not None:
            root.mkdir(parents=True, exist_ok=True)
            manager = lock_manager or LockManager(root / ".locks")
            store = atomic_store or AtomicFileStore(manager)
            self._repository = VersionedJsonRegistry(
                path=self.registry_path,
                item_loader=_load_file_factor,
                item_dumper=SavedFactor.to_dict,
                item_identity=lambda item: item.factor_id,
                legacy_items_key="factors",
                lock_manager=manager,
                atomic_store=store,
                store_name="factor_registry",
            )
            self._apply_snapshot(self._repository.load_snapshot())

    def list_factors(self) -> list[SavedFactor]:
        self._refresh()
        return sorted(self._saved.values(), key=lambda item: item.factor_id)

    def get_factor(self, factor_id: str) -> SavedFactor | None:
        self._refresh()
        saved = self._saved.get(factor_id)
        if saved is not None:
            return saved
        matches = [item for item in self._saved.values() if item.name == factor_id]
        return matches[0] if len(matches) == 1 else None

    def find_factors(
        self,
        query: str | None = None,
        *,
        include_builtins: bool = True,
    ) -> list[SavedFactor]:
        factors = self.list_factors()
        if not include_builtins:
            factors = [
                item
                for item in factors
                if not item.implementation_ref.startswith("builtin:")
            ]
        needle = str(query or "").strip()
        if not needle:
            return factors
        return [
            item
            for item in factors
            if needle in item.factor_id or needle in item.name
        ]

    def duplicate_names(self) -> dict[str, list[SavedFactor]]:
        by_name: dict[str, list[SavedFactor]] = {}
        for item in self.list_factors():
            by_name.setdefault(item.name, []).append(item)
        return {
            name: factors
            for name, factors in by_name.items()
            if len(factors) > 1
        }

    def resolve_factor_id(self, factor_id_or_name: str) -> str | None:
        saved = self.get_factor(factor_id_or_name)
        return saved.factor_id if saved is not None else None

    def save_factor(
        self,
        *,
        factor_id: str,
        name: str,
        version: str,
        implementation_ref: str,
        required_columns: tuple[str, ...],
        lookback: int,
        params: dict[str, Any] | None = None,
        input_requirements: tuple[FactorInputRequirement, ...] = (),
        created_by: str = "agent",
    ) -> SavedFactor:
        if implementation_ref.startswith("builtin:"):
            raise ValueError("built-in factors are managed by code")
        record = SavedFactor(
            factor_id=factor_id,
            name=name,
            version=version,
            implementation_ref=implementation_ref,
            required_columns=required_columns,
            lookback=lookback,
            params=params or {},
            created_by=created_by,
            created_at=shanghai_now_iso(),
            input_requirements=input_requirements,
        )
        if factor_id in _builtin_saved_factors():
            raise ValueError("built-in factors are managed by code")
        if self._repository is None:
            self._validate_factor_name(record, list(self._saved.values()))
            self._saved[factor_id] = record
            return record

        def save(items: list[SavedFactor]) -> list[SavedFactor]:
            all_current = [*_builtin_saved_factors().values(), *items]
            self._validate_factor_name(record, all_current)
            return [item for item in items if item.factor_id != factor_id] + [record]

        self._apply_snapshot(self._repository.mutate(save))
        return record

    def compute(self, factor_id: str, bars: pd.DataFrame) -> pd.Series:
        saved = self.get_factor(factor_id)
        if saved is None:
            raise ValueError(f"factor is not saved in registry: {factor_id}")
        _require_columns(bars, saved.required_columns, factor_id)
        if saved.implementation_ref.startswith("builtin:"):
            return _compute_builtin(saved.implementation_ref.removeprefix("builtin:"), bars)
        if saved.implementation_ref.startswith("file:"):
            module = _load_module(Path(saved.implementation_ref.removeprefix("file:")))
            compute = getattr(module, "compute", None)
            if not callable(compute):
                raise ValueError(f"factor file has no callable compute(): {factor_id}")
            try:
                result = compute(bars, saved.params)
            except TypeError:
                result = compute(bars)
            if not isinstance(result, pd.Series):
                raise ValueError(f"factor compute() must return pandas Series: {factor_id}")
            return result
        raise ValueError(f"unsupported factor implementation: {saved.implementation_ref}")

    def _refresh(self) -> None:
        if self._repository is not None:
            self._apply_snapshot(self._repository.load_snapshot())

    def _apply_snapshot(self, snapshot: RegistrySnapshot[SavedFactor]) -> None:
        saved = _builtin_saved_factors()
        for item in snapshot.items:
            saved[item.factor_id] = item
        self._saved = saved

    @staticmethod
    def _validate_factor_name(record: SavedFactor, current: list[SavedFactor]) -> None:
        duplicate_names = [
            item.factor_id
            for item in current
            if item.name == record.name and item.factor_id != record.factor_id
        ]
        if duplicate_names:
            raise ValueError(
                f"factor name already exists: {record.name}; use an existing factor_id "
                f"or choose a unique name. Conflicts: {duplicate_names}"
            )


def _load_file_factor(data: dict[str, Any]) -> SavedFactor:
    saved = SavedFactor.from_dict(data)
    if saved.factor_id in _builtin_saved_factors() or saved.implementation_ref.startswith(
        "builtin:"
    ):
        raise ValueError("built-in factors are managed by code")
    return saved


def _builtin_saved_factors() -> dict[str, SavedFactor]:
    builtins: dict[str, tuple[int, tuple[str, ...], tuple[FactorInputRequirement, ...]]] = {
        "momentum_20d": (20, ("symbol", "trade_date", "close"), _daily_bar_requirements("close")),
        "momentum_60d": (60, ("symbol", "trade_date", "close"), _daily_bar_requirements("close")),
        "reversal_5d": (5, ("symbol", "trade_date", "close"), _daily_bar_requirements("close")),
        "volatility_20d": (20, ("symbol", "trade_date", "close"), _daily_bar_requirements("close")),
        "turnover_20d": (
            20,
            ("symbol", "trade_date", "turnover"),
            _daily_bar_requirements("turnover"),
        ),
        "amount_zscore_20d": (
            20,
            ("symbol", "trade_date", "amount"),
            _daily_bar_requirements("amount"),
        ),
        "size_log_mktcap": (
            0,
            ("symbol", "trade_date", "total_mv"),
            _exact_daily_requirements("total_mv"),
        ),
        "pe_ttm_rank": (0, ("symbol", "trade_date", "pe_ttm"), _exact_daily_requirements("pe_ttm")),
        "pb_rank": (0, ("symbol", "trade_date", "pb"), _exact_daily_requirements("pb")),
        "dividend_yield": (
            0,
            ("symbol", "trade_date", "dv_ttm"),
            _exact_daily_requirements("dv_ttm"),
        ),
        "roe_rank": (0, ("symbol", "trade_date", "roe"), _asof_financial_requirements("roe")),
        "gross_margin_rank": (
            0,
            ("symbol", "trade_date", "gross_margin"),
            _asof_financial_requirements("gross_margin"),
        ),
        "debt_to_assets_rank": (
            0,
            ("symbol", "trade_date", "debt_to_assets"),
            _asof_financial_requirements("debt_to_assets"),
        ),
    }
    now = "builtin"
    return {
        factor_id: SavedFactor(
            factor_id=factor_id,
            name=factor_id,
            version="1.0.0",
            implementation_ref=f"builtin:{factor_id}",
            required_columns=columns,
            lookback=lookback,
            params={},
            created_by="system",
            created_at=now,
            input_requirements=requirements,
        )
        for factor_id, (lookback, columns, requirements) in builtins.items()
    }


def input_requirements_for_factor(saved: SavedFactor) -> tuple[FactorInputRequirement, ...]:
    if saved.input_requirements:
        return saved.input_requirements
    return tuple(
        _default_requirement_for_column(column)
        for column in saved.required_columns
        if column not in {"symbol", "trade_date"}
    )


def _daily_bar_requirements(field: str) -> tuple[FactorInputRequirement, ...]:
    return (
        FactorInputRequirement(
            field=field,
            target_frequency=Frequency.DAILY,
            target_calendar=TargetCalendar.TRADING_DAYS,
            entity_scope=EntityScope.STOCK_CROSS_SECTION,
            alignment_policy=AlignmentPolicy.EXACT,
            pit_required=True,
            coverage_policy=CoveragePolicy(
                min_required_field_coverage=0.80,
                min_cross_sectional_coverage=0.50,
            ),
            allowed_source_frequencies=(Frequency.DAILY,),
        ),
    )


def _exact_daily_requirements(field: str) -> tuple[FactorInputRequirement, ...]:
    return _daily_bar_requirements(field)


def _asof_financial_requirements(field: str) -> tuple[FactorInputRequirement, ...]:
    return (
        FactorInputRequirement(
            field=field,
            target_frequency=Frequency.DAILY,
            target_calendar=TargetCalendar.TRADING_DAYS,
            entity_scope=EntityScope.STOCK_CROSS_SECTION,
            alignment_policy=AlignmentPolicy.ASOF,
            pit_required=True,
            coverage_policy=CoveragePolicy(
                min_required_field_coverage=0.80,
                min_cross_sectional_coverage=0.50,
            ),
            staleness_policy=StalenessPolicy(max_staleness_days=365, p95_staleness_days=270),
            allowed_source_frequencies=(Frequency.QUARTERLY,),
        ),
    )


def _default_requirement_for_column(field: str) -> FactorInputRequirement:
    return FactorInputRequirement(
        field=field,
        target_frequency=Frequency.DAILY,
        target_calendar=TargetCalendar.TRADING_DAYS,
        entity_scope=EntityScope.STOCK_CROSS_SECTION,
        alignment_policy=AlignmentPolicy.EXACT,
        pit_required=True,
        coverage_policy=CoveragePolicy(
            min_required_field_coverage=0.80,
            min_cross_sectional_coverage=0.50,
        ),
    )

def _compute_builtin(name: str, bars: pd.DataFrame) -> pd.Series:
    if name == "momentum_20d":
        return momentum(bars, 20)
    if name == "momentum_60d":
        return momentum(bars, 60)
    if name == "reversal_5d":
        return reversal_5d(bars)
    if name == "volatility_20d":
        return volatility_20d(bars)
    if name == "turnover_20d":
        return turnover_20d(bars)
    if name == "amount_zscore_20d":
        return amount_zscore_20d(bars)
    if name == "size_log_mktcap":
        return size_log_mktcap(bars)
    if name == "pe_ttm_rank":
        return pe_ttm_rank(bars)
    if name == "pb_rank":
        return pb_rank(bars)
    if name == "dividend_yield":
        return dividend_yield(bars)
    if name == "roe_rank":
        return roe_rank(bars)
    if name == "gross_margin_rank":
        return gross_margin_rank(bars)
    if name == "debt_to_assets_rank":
        return debt_to_assets_rank(bars)
    raise ValueError(f"unsupported built-in factor: {name}")


def _load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise ValueError(f"factor implementation file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"saved_factor_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load factor implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_columns(bars: pd.DataFrame, columns: tuple[str, ...], factor_id: str) -> None:
    missing = [column for column in columns if column not in bars.columns]
    if missing:
        raise ValueError(f"factor '{factor_id}' missing required columns: {missing}")
