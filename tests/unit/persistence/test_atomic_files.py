import os
from pathlib import Path

import pytest

from qmt_agent_trader.persistence import atomic_files as atomic_files_module
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import StorageAppendRollbackError, StorageError
from qmt_agent_trader.persistence.locks import LockManager


@pytest.fixture
def atomic_store(tmp_path: Path) -> AtomicFileStore:
    return AtomicFileStore(LockManager(tmp_path / "locks"))


def test_first_jsonl_append_fsyncs_parent_directory(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, bool]] = []

    def record_directory_fsync(path: Path, *, suppress_errors: bool = True) -> None:
        calls.append((path, suppress_errors))

    monkeypatch.setattr(atomic_files_module, "_fsync_directory", record_directory_fsync)

    path = tmp_path / "events" / "stream.jsonl"
    atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert calls == [(path.parent, False)]


def test_later_jsonl_append_does_not_fsync_parent_directory(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    atomic_store.append_jsonl(path, {"event": 1}, fsync=True)
    calls: list[tuple[Path, bool]] = []

    def record_directory_fsync(path: Path, *, suppress_errors: bool = True) -> None:
        calls.append((path, suppress_errors))

    monkeypatch.setattr(atomic_files_module, "_fsync_directory", record_directory_fsync)

    atomic_store.append_jsonl(path, {"event": 2}, fsync=True)

    assert calls == []


def test_jsonl_append_without_fsync_does_not_fsync_parent_directory(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, bool]] = []

    def record_directory_fsync(path: Path, *, suppress_errors: bool = True) -> None:
        calls.append((path, suppress_errors))

    monkeypatch.setattr(atomic_files_module, "_fsync_directory", record_directory_fsync)

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
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(StorageError) as caught:
        atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert caught.value.operation == "append_jsonl"
    assert caught.value.reason == "JSONL append succeeded but directory fsync failed"
    assert caught.value.recoverable is False
    assert path.read_text() == '{"event":1}\n'


def test_failed_first_jsonl_append_restores_absent_file_state(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"

    def fail_write(_descriptor: int, _payload: bytes) -> int:
        raise OSError("injected first append failure")

    monkeypatch.setattr(os, "write", fail_write)

    with pytest.raises(StorageError, match="original state was restored"):
        atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert not path.exists()


def test_partial_first_jsonl_append_removes_new_file(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    real_write = os.write

    def partial_write(descriptor: int, payload: bytes) -> int:
        written = max(1, len(payload) // 2)
        real_write(descriptor, payload[:written])
        return written

    monkeypatch.setattr(os, "write", partial_write)

    with pytest.raises(StorageError):
        atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert not path.exists()


def test_failed_append_preserves_preexisting_empty_jsonl(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    path.parent.mkdir(parents=True)
    path.touch()

    monkeypatch.setattr(
        os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("injected")),
    )

    with pytest.raises(StorageError):
        atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert path.exists()
    assert path.read_bytes() == b""


def test_failed_first_append_reports_cleanup_failure(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    real_unlink = Path.unlink

    monkeypatch.setattr(
        os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("append failed")),
    )

    def fail_stream_cleanup(target: Path, *, missing_ok: bool = False) -> None:
        if target == path:
            raise OSError("cleanup failed")
        real_unlink(target, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_stream_cleanup)

    with pytest.raises(StorageAppendRollbackError):
        atomic_store.append_jsonl(path, {"event": 1}, fsync=True)


def test_failed_first_append_closes_descriptor_before_unlink(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    real_open = os.open
    real_close = os.close
    real_unlink = Path.unlink
    stream_descriptor: int | None = None
    closed_descriptors: set[int] = set()

    monkeypatch.setattr(
        os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("append failed")),
    )

    def capture_stream_open(
        target: Path,
        flags: int,
        mode: int = 0o777,
    ) -> int:
        nonlocal stream_descriptor
        descriptor = real_open(target, flags, mode)
        if target == path:
            stream_descriptor = descriptor
        return descriptor

    def record_close(descriptor: int) -> None:
        real_close(descriptor)
        closed_descriptors.add(descriptor)

    def assert_closed_before_unlink(target: Path, *, missing_ok: bool = False) -> None:
        if target == path:
            assert stream_descriptor is not None
            assert stream_descriptor in closed_descriptors
        real_unlink(target, missing_ok=missing_ok)

    monkeypatch.setattr(os, "open", capture_stream_open)
    monkeypatch.setattr(os, "close", record_close)
    monkeypatch.setattr(Path, "unlink", assert_closed_before_unlink)

    with pytest.raises(StorageError):
        atomic_store.append_jsonl_assume_locked(path, {"event": 1}, fsync=True)

    assert not path.exists()


def test_append_truncation_failure_raises_storage_append_rollback_error(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"event":"existing"}\n')

    monkeypatch.setattr(
        os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("append failed")),
    )
    monkeypatch.setattr(
        os,
        "ftruncate",
        lambda *_args: (_ for _ in ()).throw(OSError("truncate failed")),
    )

    with pytest.raises(StorageAppendRollbackError) as caught:
        atomic_store.append_jsonl_assume_locked(path, {"event": 1}, fsync=True)

    assert caught.value.original_append_error_type == "OSError"
    assert caught.value.rollback_error_type == "OSError"


def test_failed_first_append_reports_cleanup_directory_fsync_failure(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"

    monkeypatch.setattr(
        os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("append failed")),
    )

    def fail_directory_fsync(directory: Path, *, suppress_errors: bool = True) -> None:
        assert directory == path.parent
        assert suppress_errors is False
        assert not path.exists()
        raise OSError("cleanup directory fsync failed")

    monkeypatch.setattr(atomic_files_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(StorageAppendRollbackError) as caught:
        atomic_store.append_jsonl_assume_locked(path, {"event": 1}, fsync=True)

    assert caught.value.original_append_error_type == "OSError"
    assert caught.value.rollback_error_type == "OSError"
    assert not path.exists()


def test_failed_first_append_strictly_fsyncs_parent_after_cleanup(
    atomic_store: AtomicFileStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events" / "stream.jsonl"
    calls: list[tuple[Path, bool, bool]] = []

    monkeypatch.setattr(
        os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("append failed")),
    )

    def record_directory_fsync(directory: Path, *, suppress_errors: bool = True) -> None:
        calls.append((directory, suppress_errors, path.exists()))

    monkeypatch.setattr(atomic_files_module, "_fsync_directory", record_directory_fsync)

    with pytest.raises(StorageError):
        atomic_store.append_jsonl(path, {"event": 1}, fsync=True)

    assert calls == [(path.parent, False, False)]
    assert not path.exists()
