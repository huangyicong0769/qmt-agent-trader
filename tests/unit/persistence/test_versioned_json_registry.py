from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import (
    StorageCorruptError,
    StorageError,
    StorageRevisionConflictError,
    StorageSchemaMismatchError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.repositories.versioned_json import VersionedJsonRegistry


@dataclass(frozen=True)
class _Item:
    identity: str
    value: int


def _load_item(payload: dict[str, Any]) -> _Item:
    if set(payload) != {"identity", "value"}:
        raise ValueError("invalid item fields")
    return _Item(identity=str(payload["identity"]), value=int(payload["value"]))


def _dump_item(item: _Item) -> dict[str, Any]:
    return {"identity": item.identity, "value": item.value}


def _repository(
    path: Path,
    *,
    store: AtomicFileStore | None = None,
) -> VersionedJsonRegistry[_Item]:
    manager = LockManager(path.parent / "locks", timeout_seconds=2)
    return VersionedJsonRegistry(
        path=path,
        item_loader=_load_item,
        item_dumper=_dump_item,
        item_identity=lambda item: item.identity,
        lock_manager=manager,
        atomic_store=store or AtomicFileStore(manager),
        store_name="test_registry",
    )


def test_versioned_registry_writes_and_verifies_exact_v2_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    repository = _repository(path)

    snapshot = repository.mutate(lambda items: [*items, _Item("a", 1)])

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "revision",
        "updated_at",
        "content_hash",
        "items",
    }
    assert payload["schema_version"] == 2
    assert payload["revision"] == 1
    assert len(payload["content_hash"]) == 64
    assert snapshot.items == (_Item("a", 1),)
    assert repository.load_snapshot() == snapshot


def test_versioned_registry_rejects_stale_expected_revision(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "registry.json")
    repository.mutate(lambda items: [*items, _Item("a", 1)])

    with pytest.raises(StorageRevisionConflictError):
        repository.mutate(
            lambda items: [*items, _Item("b", 2)],
            expected_revision=0,
        )

    assert [item.identity for item in repository.load_snapshot().items] == ["a"]


def test_versioned_registry_rejects_duplicate_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "registry.json")

    with pytest.raises(StorageValidationError, match="duplicate"):
        repository.mutate(lambda _items: [_Item("a", 1), _Item("a", 2)])


def test_versioned_registry_rejects_v1_without_modifying_it(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    original = {"version": 1, "records": [{"identity": "old", "value": 7}]}
    path.write_text(json.dumps(original), encoding="utf-8")
    repository = _repository(path)

    with pytest.raises(StorageSchemaMismatchError):
        repository.load_snapshot()

    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_versioned_registry_rejects_corrupt_and_hash_invalid_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(StorageCorruptError):
        _repository(path).load_snapshot()

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "revision": 1,
                "updated_at": "2026-07-11T00:00:00+08:00",
                "content_hash": "0" * 64,
                "items": [{"identity": "a", "value": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StorageCorruptError, match="hash"):
        _repository(path).load_snapshot()


def test_versioned_registry_rejects_unknown_schema_without_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    payload = {"schema_version": 3, "items": []}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StorageSchemaMismatchError):
        _repository(path).load_snapshot()

    assert json.loads(path.read_text(encoding="utf-8")) == payload


class _FailBeforeReplaceStore(AtomicFileStore):
    def write_bytes(
        self,
        path: Path,
        content: bytes,
        *,
        create_only: bool = False,
        validator: Any = None,
        fault_hook: Any = None,
    ) -> None:
        def fail(_stage: str, _temp: Path) -> None:
            raise OSError("injected before replace")

        super().write_bytes(
            path,
            content,
            create_only=create_only,
            validator=validator,
            fault_hook=fail,
        )


def test_versioned_registry_fault_before_replace_preserves_previous_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    repository = _repository(path)
    original = repository.mutate(lambda items: [*items, _Item("a", 1)])
    manager = repository.lock_manager
    failing = _repository(path, store=_FailBeforeReplaceStore(manager))

    with pytest.raises(StorageError, match="atomic write failed"):
        failing.mutate(lambda items: [*items, _Item("b", 2)])

    assert repository.load_snapshot() == original
