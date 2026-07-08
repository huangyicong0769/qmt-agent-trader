from __future__ import annotations

import pandas as pd

from qmt_agent_trader.cli import main as cli
from qmt_agent_trader.data.storage import DataLake


def test_migrate_new_layout_moves_stable_and_batch_sources_to_registry_path(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.0},
                {"ts_code": "000002.SZ", "trade_date": "20240102", "close": 20.0},
            ]
        ),
        "raw",
        "tushare_daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.5},
                {"ts_code": "000003.SZ", "trade_date": "20240103", "close": 30.0},
            ]
        ),
        "raw",
        "tushare_daily_20240102_20240103",
    )
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000004.SZ", "trade_date": "20240104", "close": 40.0}]),
        "raw",
        "tushare_daily_adjusted",
    )
    monkeypatch.setattr(cli, "_data_lake", lambda: lake)

    cli.data_migrate_new_layout(keep_legacy=False)

    migrated = lake.read_parquet("raw", "tushare/daily").sort_values(
        ["ts_code", "trade_date"]
    )
    assert migrated.to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.5},
        {"ts_code": "000002.SZ", "trade_date": "20240102", "close": 20.0},
        {"ts_code": "000003.SZ", "trade_date": "20240103", "close": 30.0},
    ]
    assert not lake.dataset_path("raw", "tushare_daily").exists()
    assert not lake.dataset_path("raw", "tushare_daily_20240102_20240103").exists()
    assert lake.dataset_path("raw", "tushare_daily_adjusted").exists()
