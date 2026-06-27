from __future__ import annotations

from qmt_agent_trader.agent.tools import cache


def test_cached_validation_without_freshness_fields_is_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    cache.put_cached_validation(
        "factor_a",
        "20240101",
        "20240131",
        {"status": "validated", "name": "factor_a"},
    )

    assert cache.get_cached_validation("factor_a", "20240101", "20240131") is None


def test_cached_validation_with_freshness_fields_is_returned(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    result = {
        "status": "validated",
        "name": "factor_a",
        "actual_data_start": "20240102",
        "actual_data_end": "20240131",
        "data_freshness": "covers_requested_end",
    }
    cache.put_cached_validation("factor_a", "20240101", "20240131", result)

    cached = cache.get_cached_validation("factor_a", "20240101", "20240131")

    assert cached == result
