"""Saved factor registry for built-in and agent-authored research factors."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from qmt_agent_trader.core.ids import shanghai_now_iso
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
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["required_columns"] = list(self.required_columns)
        return data


class FactorRegistry:
    """Registry of saved factors.

    Built-in factors and agent-saved factors share the same public lookup and
    compute path. Draft files are intentionally invisible until saved here.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.registry_path = root / "registry.json" if root is not None else None
        self._saved = _builtin_saved_factors()
        if root is not None and self.registry_path is not None:
            root.mkdir(parents=True, exist_ok=True)
            self._saved.update(self._load_file_registry())

    def list_factors(self) -> list[SavedFactor]:
        return sorted(self._saved.values(), key=lambda item: item.factor_id)

    def get_factor(self, factor_id: str) -> SavedFactor | None:
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
        created_by: str = "agent",
    ) -> SavedFactor:
        if implementation_ref.startswith("builtin:"):
            raise ValueError("built-in factors are managed by code")
        duplicate_names = [
            item.factor_id
            for item in self._saved.values()
            if item.name == name and item.factor_id != factor_id
        ]
        if duplicate_names:
            raise ValueError(
                f"factor name already exists: {name}; use an existing factor_id "
                f"or choose a unique name. Conflicts: {duplicate_names}"
            )
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
        )
        self._saved[factor_id] = record
        self._persist_file_registry()
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

    def _load_file_registry(self) -> dict[str, SavedFactor]:
        if self.registry_path is None or not self.registry_path.exists():
            return {}
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        factors = payload.get("factors", [])
        if not isinstance(factors, list):
            return {}
        loaded: dict[str, SavedFactor] = {}
        for item in factors:
            if isinstance(item, dict):
                saved = SavedFactor.from_dict(item)
                loaded[saved.factor_id] = saved
        return loaded

    def _persist_file_registry(self) -> None:
        if self.registry_path is None:
            return
        file_factors = [
            item.to_dict()
            for item in self.list_factors()
            if not item.implementation_ref.startswith("builtin:")
        ]
        payload = {"version": 1, "factors": file_factors}
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def _builtin_saved_factors() -> dict[str, SavedFactor]:
    builtins = {
        "momentum_20d": (20, ("symbol", "trade_date", "close")),
        "momentum_60d": (60, ("symbol", "trade_date", "close")),
        "reversal_5d": (5, ("symbol", "trade_date", "close")),
        "volatility_20d": (20, ("symbol", "trade_date", "close")),
        "turnover_20d": (20, ("symbol", "trade_date", "turnover")),
        "amount_zscore_20d": (20, ("symbol", "trade_date", "amount")),
        "size_log_mktcap": (0, ("symbol", "trade_date", "total_mv")),
        "pe_ttm_rank": (0, ("symbol", "trade_date", "pe_ttm")),
        "pb_rank": (0, ("symbol", "trade_date", "pb")),
        "dividend_yield": (0, ("symbol", "trade_date", "dv_ttm")),
        "roe_rank": (0, ("symbol", "trade_date", "roe")),
        "gross_margin_rank": (0, ("symbol", "trade_date", "gross_margin")),
        "debt_to_assets_rank": (0, ("symbol", "trade_date", "debt_to_assets")),
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
        )
        for factor_id, (lookback, columns) in builtins.items()
    }

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
