from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.todos import TodoListStore
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.errors import (
    StorageCorruptError,
    StorageError,
    StorageRevisionConflictError,
    StorageValidationError,
)
from qmt_agent_trader.universe.builtins import broad_universe_spec
from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.registry import UniverseRegistry
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.schemas import ChatSession


def _add_todo(root: str, session_id: str, title: str) -> None:
    TodoListStore(Path(root)).add_item(session_id, title=title)


def _update_todo(root: str, session_id: str, item_id: str, notes: str) -> None:
    TodoListStore(Path(root)).update_item(session_id, item_id, notes=notes)


def _append_artifact(root: str, experiment_id: str, artifact: str) -> None:
    ExperimentStore(Path(root)).add_artifact(experiment_id, artifact)


def _append_lesson(root: str, experiment_id: str, lesson: str) -> None:
    ExperimentStore(Path(root)).add_lesson(experiment_id, lesson)


def _save_universe(root: str, payload: dict[str, object]) -> None:
    UniverseRegistry(Path(root)).save(UniverseSpec.model_validate(payload))


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


def test_concurrent_updates_to_distinct_existing_todos_are_both_retained(tmp_path: Path) -> None:
    root = tmp_path / "todos"
    store = TodoListStore(root)
    initial = store.replace_items("chat_1", [{"title": "one"}, {"title": "two"}])
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_update_todo, str(root), "chat_1", item.item_id, note)
            for item, note in zip(initial.items, ("first", "second"), strict=True)
        ]
        for future in futures:
            future.result()
    assert [item.notes for item in store.get("chat_1").items] == ["first", "second"]


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


def test_concurrent_experiment_lesson_event_appends_are_retained(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    experiment = store.create_experiment("factor")
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_append_lesson, str(root), experiment.experiment_id, lesson)
            for lesson in ("event-a", "event-b")
        ]
        for future in futures:
            future.result()
    assert set(store.get_experiment(experiment.experiment_id).lessons) == {
        "event-a", "event-b"
    }


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


def test_todo_stale_revision_is_rejected(tmp_path: Path) -> None:
    store = TodoListStore(tmp_path / "todos")
    current = store.replace_items("chat_1", [{"title": "one"}])
    store.add_item("chat_1", title="two", expected_revision=current.revision)
    with pytest.raises(StorageRevisionConflictError):
        store.add_item("chat_1", title="stale", expected_revision=current.revision)


def test_chat_legacy_ui_record_migrates_idempotently_and_is_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    legacy = {
        "sid": "s7",
        "name": "Legacy",
        "counter": 9,
        "preview": "hello",
        "messages": [{"role": "user", "content": "hello", "metadata": {"x": 1}}],
    }
    (root / "s7.json").write_text(json.dumps(legacy), encoding="utf-8")
    repository = ChatSessionRepository(root)
    monkeypatch.chdir(tmp_path / "other") if (tmp_path / "other").mkdir() is None else None
    first = repository.get("s7")
    second = repository.get("s7")
    assert first is not None and second is not None
    assert first.session_id == "s7" and first.title == "Legacy"
    assert first.context["legacy_ui"] == {"counter": 9, "preview": "hello"}
    assert first.messages[0].metadata == {"x": 1}
    assert second.revision == first.revision == 1


def test_chat_stale_revision_is_rejected(tmp_path: Path) -> None:
    repository = ChatSessionRepository(tmp_path / "sessions")
    first = repository.create(ChatSession(session_id="chat_1"))
    repository.update(
        "chat_1", lambda session: session.model_copy(update={"title": "new"}),
        expected_revision=first.revision,
    )
    with pytest.raises(StorageRevisionConflictError):
        repository.update("chat_1", lambda session: session,
            expected_revision=first.revision)


def test_universe_concurrent_same_and_distinct_ids_remain_valid(tmp_path: Path) -> None:
    root = tmp_path / "universes"
    first = broad_universe_spec("stock")
    changed = first.model_copy(update={"description": "concurrent update"})
    distinct = broad_universe_spec("etf")
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_save_universe, str(root), spec.model_dump(mode="json"))
            for spec in (first, changed, distinct)
        ]
        for future in futures:
            future.result()
    registry = UniverseRegistry(root)
    assert {item.universe_id for item in registry.list()} == {
        first.universe_id,
        distinct.universe_id,
    }
    stored = registry.repository.load(first.universe_id)
    assert stored.revision == 2


def test_universe_stale_revision_is_rejected(tmp_path: Path) -> None:
    registry = UniverseRegistry(tmp_path / "universes")
    spec = broad_universe_spec("stock")
    registry.save(spec, expected_revision=0)
    with pytest.raises(StorageRevisionConflictError):
        registry.save(spec, expected_revision=0)


def test_invalid_hash_and_schema_are_structured_errors(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    experiment = store.create_experiment("factor")
    path = store._path(experiment.experiment_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StorageCorruptError):
        store.get_experiment(experiment.experiment_id)

    payload["schema_version"] = 3
    payload["content_hash"] = store.repository._hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StorageValidationError):
        store.get_experiment(experiment.experiment_id)


def test_fault_before_replace_preserves_previous_record(tmp_path: Path) -> None:
    store = TodoListStore(tmp_path / "todos")
    original = store.replace_items("chat_1", [{"title": "one"}])

    def fail(stage: str, _path: Path) -> None:
        if stage == "before_replace":
            raise RuntimeError("injected fault")

    store.repository.fault_hook = fail
    with pytest.raises(StorageError):
        store.add_item("chat_1", title="lost")
    store.repository.fault_hook = None
    restored = store.get("chat_1")
    assert restored.revision == original.revision
    assert [item.title for item in restored.items] == ["one"]


def test_universe_previous_root_migrates_to_canonical_idempotently(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    previous = UniverseRegistry(tmp_path / "universes" / "registry")
    spec = broad_universe_spec("stock")
    previous.save(spec)
    source = previous.path_for(spec.universe_id)
    canonical = UniverseRegistry.for_lake(lake)
    migrated = canonical.load_record(spec.universe_id)
    assert migrated is not None and migrated.spec == spec
    assert source.exists()
    again = UniverseRegistry.for_lake(lake).load_record(spec.universe_id)
    assert again is not None and again.revision == migrated.revision == 1
