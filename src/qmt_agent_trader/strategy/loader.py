"""Safe loading helpers for generated strategy code."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pandas as pd
from pydantic import BaseModel

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.strategy.base import StrategyContext
from qmt_agent_trader.strategy.models import StrategySpec, strategy_spec_from_agent_spec


class StrategyLoadError(Exception):
    """Raised when generated strategy code cannot be loaded safely."""


class LoadedStrategy(BaseModel):
    strategy_id: str
    version: str
    module_path: str
    spec: StrategySpec | None = None
    callable_name: str = "generate_signals"


def static_check_strategy_code(code: str) -> list[str]:
    return CodeSandbox().static_scan_code(code)


def static_check_strategy_file(path: Path) -> list[str]:
    if not path.exists():
        return [f"strategy file not found: {path}"]
    return static_check_strategy_code(path.read_text(encoding="utf-8"))


def load_strategy_from_file(
    path: Path,
    *,
    allowed_roots: list[Path] | None = None,
) -> LoadedStrategy:
    resolved = path.resolve()
    if allowed_roots is not None and not any(
        _is_under(resolved, root.resolve()) for root in allowed_roots
    ):
        raise StrategyLoadError(f"strategy path is outside allowed roots: {resolved}")
    issues = static_check_strategy_file(resolved)
    if issues:
        raise StrategyLoadError("static scan failed: " + "; ".join(issues))
    module = _load_module(resolved)
    callable_name = _resolve_callable_name(module)
    spec = _extract_spec(module, resolved)
    return LoadedStrategy(
        strategy_id=spec.strategy_id if spec else resolved.parent.name,
        version=spec.version if spec else "0.1.0",
        module_path=str(resolved),
        spec=spec,
        callable_name=callable_name,
    )


def run_strategy_generate_signals(
    loaded: LoadedStrategy,
    context: StrategyContext,
) -> pd.DataFrame:
    module = _load_module(Path(loaded.module_path))
    func = getattr(module, loaded.callable_name, None)
    if not callable(func):
        raise StrategyLoadError(f"callable not found: {loaded.callable_name}")
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    if params and params[0].name in {"data", "frame", "bars"}:
        result = func(context.bars)
    else:
        result = func(context)
    if not isinstance(result, pd.DataFrame):
        raise StrategyLoadError("generate_signals must return a pandas DataFrame")
    return result


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"strategy_{path.parent.name}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"unable to load strategy module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_callable_name(module: ModuleType) -> str:
    if callable(getattr(module, "generate_signals", None)):
        return "generate_signals"
    factory = getattr(module, "strategy_factory", None)
    if callable(factory):
        strategy = factory({})
        if callable(getattr(strategy, "generate_signals", None)):
            return "strategy_factory"
    raise StrategyLoadError("strategy must expose generate_signals() or strategy_factory()")


def _extract_spec(module: ModuleType, path: Path) -> StrategySpec | None:
    raw = getattr(module, "STRATEGY_SPEC", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StrategyLoadError("STRATEGY_SPEC must be a dict")
    try:
        return strategy_spec_from_agent_spec(raw)
    except Exception as exc:
        raise StrategyLoadError(f"invalid STRATEGY_SPEC in {path}: {exc}") from exc


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
