from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any

from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySource, StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def _save_factor(root: str, factor_id: str, queue: Any, barrier: Any) -> None:
    registry = FactorRegistry(Path(root))
    barrier.wait()
    try:
        registry.save_factor(
            factor_id=factor_id,
            name=factor_id,
            version="0.1.0",
            implementation_ref=f"file:/tmp/{factor_id}.py",
            required_columns=("close",),
            lookback=1,
        )
        queue.put((factor_id, "saved"))
    except Exception as exc:
        queue.put((factor_id, type(exc).__name__))


def _save_factor_version(
    root: str,
    factor_id: str,
    version: str,
    queue: Any,
    barrier: Any,
) -> None:
    registry = FactorRegistry(Path(root))
    barrier.wait()
    try:
        registry.save_factor(
            factor_id=factor_id,
            name=factor_id,
            version=version,
            implementation_ref=f"file:/tmp/{factor_id}-{version}.py",
            required_columns=("close",),
            lookback=1,
        )
        queue.put((version, "saved"))
    except Exception as exc:
        queue.put((version, type(exc).__name__))


def _save_strategy(root: str, strategy_id: str, queue: Any, barrier: Any) -> None:
    registry = StrategyRegistry(Path(root))
    barrier.wait()
    try:
        registry.save_candidate(
            SavedStrategy(
                strategy_id=strategy_id,
                name=strategy_id,
                version="0.1.0",
                source=StrategySource.AGENT_GENERATED,
                status=ApprovalStatus.GENERATED_BY_LLM,
                spec=StrategySpec(strategy_id=strategy_id, name=strategy_id),
                implementation_ref=f"file:/tmp/{strategy_id}.py",
            )
        )
        queue.put((strategy_id, "saved"))
    except Exception as exc:
        queue.put((strategy_id, type(exc).__name__))


def _run_two(target: Any, root: Path, identities: tuple[str, str]) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    barrier = context.Barrier(2)
    processes = [
        context.Process(target=target, args=(str(root), identity, queue, barrier))
        for identity in identities
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    return results


def _run_factor_versions(root: Path) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_save_factor_version,
            args=(str(root), "factor_same", version, queue, barrier),
        )
        for version in ("1.0.0", "2.0.0")
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    return results


def test_factor_registry_two_processes_preserve_distinct_additions(tmp_path: Path) -> None:
    root = tmp_path / "factors"

    results = _run_two(_save_factor, root, ("factor_a", "factor_b"))

    assert sorted(results) == [("factor_a", "saved"), ("factor_b", "saved")]
    assert [
        item.factor_id
        for item in FactorRegistry(root).list_factors()
        if not item.implementation_ref.startswith("builtin:")
    ] == ["factor_a", "factor_b"]


def test_strategy_registry_two_processes_preserve_distinct_additions(tmp_path: Path) -> None:
    root = tmp_path / "strategies"

    results = _run_two(_save_strategy, root, ("strategy_a", "strategy_b"))

    assert sorted(results) == [("strategy_a", "saved"), ("strategy_b", "saved")]
    assert [
        item.strategy_id
        for item in StrategyRegistry(root).list_strategies(include_builtins=False)
    ] == ["strategy_a", "strategy_b"]


def test_factor_registry_same_id_race_converges_without_duplicate_records(tmp_path: Path) -> None:
    root = tmp_path / "factors"

    results = _run_two(_save_factor, root, ("factor_same", "factor_same"))

    assert sorted(status for _, status in results) == ["saved", "saved"]
    assert json.loads((root / "registry.json").read_text())["revision"] == 1
    assert [
        item.factor_id
        for item in FactorRegistry(root).list_factors()
        if not item.implementation_ref.startswith("builtin:")
    ] == ["factor_same"]


def test_factor_registry_divergent_same_id_race_has_structured_conflict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "factors"

    results = _run_factor_versions(root)

    assert sorted(status for _, status in results) == [
        "StorageRevisionConflictError",
        "saved",
    ]
    persisted = FactorRegistry(root).get_factor("factor_same")
    assert persisted is not None
    assert persisted.version in {"1.0.0", "2.0.0"}
    assert json.loads((root / "registry.json").read_text())["revision"] == 1


def test_strategy_registry_same_id_race_has_one_winner(tmp_path: Path) -> None:
    root = tmp_path / "strategies"

    results = _run_two(_save_strategy, root, ("strategy_same", "strategy_same"))

    assert sorted(status for _, status in results) == ["ValueError", "saved"]
    assert [
        item.strategy_id
        for item in StrategyRegistry(root).list_strategies(include_builtins=False)
    ] == ["strategy_same"]
