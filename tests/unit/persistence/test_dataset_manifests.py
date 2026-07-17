from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence import dataset_manifests


def test_governed_write_creates_dataset_manifest(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    path = lake.write_parquet(
        pd.DataFrame([{"value": 1}]),
        "raw",
        "fixture",
    )

    manifest_path = dataset_manifests.dataset_manifest_path(path)

    assert manifest_path.exists()
    first = lake.dataset_fingerprint("raw", "fixture")
    assert first is not None


def test_second_fingerprint_uses_manifest_without_reading_payload(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"value": 1}]),
        "raw",
        "fixture",
    )
    first = lake.dataset_fingerprint("raw", "fixture")
    monkeypatch.setattr(
        dataset_manifests,
        "_content_digest",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("payload must not be rehashed")
        ),
    )

    second = lake.dataset_fingerprint("raw", "fixture")

    assert second == first


def test_same_shape_rewrite_changes_manifest_fingerprint(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"value": 1}]),
        "raw",
        "fixture",
    )
    first = lake.dataset_fingerprint("raw", "fixture")

    lake.write_parquet(
        pd.DataFrame([{"value": 2}]),
        "raw",
        "fixture",
    )
    second = lake.dataset_fingerprint("raw", "fixture")

    assert first != second
