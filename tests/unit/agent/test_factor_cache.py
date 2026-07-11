from __future__ import annotations

from qmt_agent_trader.agent.tools import cache
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.cache import ContentAddressedCache
from qmt_agent_trader.persistence.locks import LockManager


def _store(tmp_path):
    return ContentAddressedCache(
        tmp_path / "cache", AtomicFileStore(LockManager(tmp_path / "locks"))
    )


def test_cached_validation_without_freshness_fields_is_invalidated_through_cache_api(
    tmp_path,
) -> None:
    warnings = []
    store = ContentAddressedCache(
        tmp_path / "cache",
        AtomicFileStore(LockManager(tmp_path / "locks")),
        warning_sink=warnings.append,
    )
    cache.put_cached_validation(
        "factor_a",
        "20240101",
        "20240131",
        {"status": "validated", "name": "factor_a"},
        store,
    )

    assert cache.get_cached_validation("factor_a", "20240101", "20240131", store) is None
    assert store.metrics["invalidations"] == 1
    assert warnings[-1]["reason"] == "CACHE_VALIDATION_STALE"


def test_cached_validation_with_freshness_fields_is_returned(tmp_path) -> None:
    store = _store(tmp_path)
    result = {
        "status": "validated",
        "name": "factor_a",
        "actual_data_start": "20240102",
        "actual_data_end": "20240131",
        "data_freshness": "covers_requested_end",
    }
    cache.put_cached_validation("factor_a", "20240101", "20240131", result, store)

    cached = cache.get_cached_validation("factor_a", "20240101", "20240131", store)

    assert cached == result
