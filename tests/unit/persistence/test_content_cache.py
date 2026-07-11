from __future__ import annotations

import json
import multiprocessing
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.cache import ContentAddressedCache
from qmt_agent_trader.persistence.locks import LockManager


def _cache_writer(root: str, locks: str, key: str, worker: int) -> None:
    cache = ContentAddressedCache(Path(root), AtomicFileStore(LockManager(Path(locks))))
    for _ in range(20):
        cache.put("concurrent", key, {"worker": worker})


def _cache(
    tmp_path: Path, now: datetime, warnings: list[dict[str, object]]
) -> ContentAddressedCache:
    return ContentAddressedCache(
        tmp_path / "injected-cache",
        AtomicFileStore(LockManager(tmp_path / "locks")),
        ttl=timedelta(seconds=10),
        clock=lambda: now,
        warning_sink=warnings.append,
    )


def test_cache_hit_miss_ttl_and_content_addressed_key(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    cache = _cache(tmp_path, now, [])
    key = cache.key_for({"b": 2, "a": 1})
    assert key == cache.key_for({"a": 1, "b": 2})
    assert cache.get("factor", key) is None
    cache.put("factor", key, {"status": "ok"})
    assert cache.get("factor", key) == {"status": "ok"}
    cache.clock = lambda: now + timedelta(seconds=11)
    assert cache.get("factor", key) is None


def test_corrupt_cache_invalidates_with_warning_and_never_blocks(tmp_path: Path) -> None:
    warnings: list[dict[str, object]] = []
    cache = _cache(tmp_path, datetime(2026, 1, 1, tzinfo=UTC), warnings)
    key = cache.key_for({"factor": "x"})
    path = cache.path_for("factor", key)
    path.parent.mkdir(parents=True)
    path.write_text("not-json")
    assert cache.get("factor", key) is None
    assert not path.exists()
    assert warnings[-1]["reason"] == "CACHE_CORRUPT_INVALIDATED"
    assert cache.metrics["corrupt_invalidations"] == 1


def test_warning_sink_failure_never_blocks_cache_miss(tmp_path: Path) -> None:
    cache = ContentAddressedCache(
        tmp_path / "cache",
        AtomicFileStore(LockManager(tmp_path / "locks")),
        warning_sink=lambda payload: (_ for _ in ()).throw(RuntimeError(str(payload))),
    )
    key = cache.key_for({"factor": "x"})
    path = cache.path_for("factor", key)
    path.parent.mkdir(parents=True)
    path.write_text("not-json")

    assert cache.get("factor", key) is None


def test_cache_unlink_failure_warns_and_never_blocks_research(
    tmp_path: Path, monkeypatch
) -> None:
    warnings: list[dict[str, object]] = []
    cache = _cache(tmp_path, datetime(2026, 1, 1, tzinfo=UTC), warnings)
    key = cache.key_for({"factor": "unlink-failure"})
    path = cache.path_for("factor", key)
    path.parent.mkdir(parents=True)
    path.write_text("not-json")
    original_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self == path:
            raise OSError("read-only cache")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    assert cache.get("factor", key) is None
    assert cache.metrics["invalidation_failures"] == 1
    assert any(item["reason"] == "CACHE_INVALIDATION_FAILED" for item in warnings)


def test_corrupt_reader_cannot_unlink_concurrent_valid_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    cache = _cache(tmp_path, datetime(2026, 1, 1, tzinfo=UTC), [])
    key = cache.key_for({"factor": "race"})
    path = cache.path_for("factor", key)
    path.parent.mkdir(parents=True)
    path.write_text("not-json")
    read_started = threading.Event()
    allow_read = threading.Event()
    original_read_text = Path.read_text

    def paused_read_text(self, *args, **kwargs):
        value = original_read_text(self, *args, **kwargs)
        if self == path and value == "not-json":
            read_started.set()
            assert allow_read.wait(timeout=5)
        return value

    monkeypatch.setattr(Path, "read_text", paused_read_text)
    result: list[dict[str, object] | None] = []
    reader = threading.Thread(target=lambda: result.append(cache.get("factor", key)))
    reader.start()
    assert read_started.wait(timeout=5)

    writer_done = threading.Event()

    def replace_value() -> None:
        cache.put("factor", key, {"version": 2})
        writer_done.set()

    writer = threading.Thread(target=replace_value)
    writer.start()
    writer_completed_during_read = writer_done.wait(timeout=0.2)
    allow_read.set()
    reader.join(timeout=5)
    writer.join(timeout=5)

    assert writer_completed_during_read is False
    assert result == [None]
    assert cache.get("factor", key) == {"version": 2}


def test_failed_cache_write_preserves_previous_value(tmp_path: Path) -> None:
    cache = _cache(tmp_path, datetime(2026, 1, 1, tzinfo=UTC), [])
    key = cache.key_for({"factor": "x"})
    cache.put("factor", key, {"version": 1})

    with pytest.raises(RuntimeError):
        cache.put(
            "factor",
            key,
            {"version": 2},
            fault_hook=lambda stage, path: (_ for _ in ()).throw(RuntimeError(stage)),
            raise_on_error=True,
        )
    assert json.loads(cache.path_for("factor", key).read_text())["value"] == {"version": 1}


def test_cache_root_is_injected_and_independent_of_cwd(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "canonical"
    cache = ContentAddressedCache(
        root,
        AtomicFileStore(LockManager(tmp_path / "locks")),
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    key = cache.key_for({"x": 1})
    cache.put("tool", key, {"ok": True})
    assert cache.path_for("tool", key).is_relative_to(root)


def test_concurrent_cache_writes_never_expose_partial_json(tmp_path: Path) -> None:
    cache = _cache(tmp_path, datetime(2026, 1, 1, tzinfo=UTC), [])
    key = cache.key_for({"same": "request"})
    processes = [
        multiprocessing.Process(
            target=_cache_writer,
            args=(str(cache.root), str(tmp_path / "locks"), key, worker),
        )
        for worker in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert cache.get("concurrent", key) in tuple({"worker": i} for i in range(4))
