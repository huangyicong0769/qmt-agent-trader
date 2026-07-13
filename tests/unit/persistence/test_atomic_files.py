from pathlib import Path

import pytest

from qmt_agent_trader.persistence import atomic_files as atomic_files_module
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import StorageError
from qmt_agent_trader.persistence.locks import LockManager


@pytest.fixture
def atomic_store(tmp_path: Path) -> AtomicFileStore:
    return AtomicFileStore(LockManager(tmp_path / "locks"))


def test_first_jsonl_append_fsyncs_parent_directory(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(atomic_files_module, "_fsync_directory", calls.append)

    path = tmp_path / "events" / "stream.jsonl"
    atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert calls == [path.parent]


def test_later_jsonl_append_does_not_fsync_parent_directory(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    atomic_store.append_jsonl(path, {"event": 1}, fsync=True)
    calls: list[Path] = []
    monkeypatch.setattr(atomic_files_module, "_fsync_directory", calls.append)

    atomic_store.append_jsonl(path, {"event": 2}, fsync=True)

    assert calls == []


def test_jsonl_append_without_fsync_does_not_fsync_parent_directory(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(atomic_files_module, "_fsync_directory", calls.append)

    atomic_store.append_jsonl(
        tmp_path / "events" / "stream.jsonl", {"event": 1}, fsync=False
    )

    assert calls == []


def test_directory_fsync_failure_reports_uncertain_durability_without_rollback(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"

    def fail_directory_fsync(directory: Path) -> None:
        raise OSError("directory fsync failed")

    monkeypatch.setattr(atomic_files_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(StorageError) as caught:
        atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert caught.value.operation == "append_jsonl"
    assert caught.value.reason == "JSONL append succeeded but directory fsync failed"
    assert caught.value.recoverable is False
    assert path.read_text() == '{"event":1}\n'
