from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.todos import TodoListStore
from qmt_agent_trader.persistence.errors import StorageCorruptError, StorageValidationError
from qmt_agent_trader.universe.registry import UniverseRegistry


def _add_todo(root: str, session_id: str, title: str) -> None:
    TodoListStore(Path(root)).add_item(session_id, title=title)


def _append_artifact(root: str, experiment_id: str, artifact: str) -> None:
    ExperimentStore(Path(root)).add_artifact(experiment_id, artifact)


def test_concurrent_todo_updates_retain_distinct_items(tmp_path: Path) -> None:
    root = tmp_path / "todos"
    store = TodoListStore(root)
    initial = store.replace_items("chat_1", [{"title": "one"}, {"title": "two"}])
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_add_todo, str(root), "chat_1", title) for title in ("three", "four")
        ]
        for future in futures:
            future.result()
    assert {item.title for item in store.get("chat_1").items} == {"one", "two", "three", "four"}
    assert store.get("chat_1").revision == initial.revision + 2


def test_concurrent_experiment_appends_retain_every_artifact(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    experiment = store.create_experiment("factor")
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_append_artifact, str(root), experiment.experiment_id, value)
            for value in ("a", "b")
        ]
        for future in futures:
            future.result()
    assert set(store.get_experiment(experiment.experiment_id).artifacts) == {"a", "b"}


def test_corruption_is_diagnostic_and_explicitly_quarantined(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    experiment = store.create_experiment("factor")
    store._path(experiment.experiment_id).write_text("{broken", encoding="utf-8")
    records = store.search_experiments()
    assert records == []
    assert len(store.last_diagnostics) == 1
    assert isinstance(store.last_diagnostics[0].error, StorageCorruptError)
    target = store.repository.quarantine(experiment.experiment_id)
    assert target.exists()
    assert not store._path(experiment.experiment_id).exists()


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", "a\\b", ".."])
def test_universe_ids_reject_path_traversal(tmp_path: Path, unsafe: str) -> None:
    registry = UniverseRegistry(tmp_path)
    with pytest.raises(StorageValidationError):
        registry.path_for(unsafe)
