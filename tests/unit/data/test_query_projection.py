from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.query_projection import (
    build_research_feature_frame,
    load_daily_market,
    load_financial_snapshot,
    load_macro_snapshot,
)
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.table_builder import DataTableBuilder


def test_load_daily_market_uses_new_registry_dataset_only(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 1.0,
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                }
            ]
        ),
        "raw",
        "tushare_daily",
    )

    assert load_daily_market(lake, symbols=["000001.SZ"]).empty

    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 1.0,
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )

    frame = load_daily_market(lake, symbols=["000001.SZ"])

    assert frame["symbol"].tolist() == ["000001.SZ"]
    assert frame["close"].tolist() == [1.1]


def test_financial_snapshot_and_feature_frame_are_point_in_time(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240403",
                    "open": 2.0,
                    "high": 2.0,
                    "low": 2.0,
                    "close": 2.0,
                },
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240401",
                    "f_ann_date": "20240402",
                    "end_date": "20231231",
                    "roe": 9.5,
                }
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )
    DataTableBuilder(lake).build("financial_reports_wide")

    before = load_financial_snapshot(lake, as_of_date="20240401", symbols=["000001.SZ"])
    after = load_financial_snapshot(lake, as_of_date="20240403", symbols=["000001.SZ"])
    features = build_research_feature_frame(
        lake,
        symbols=["000001.SZ"],
        start="20240101",
        end="20240405",
    )

    assert before.empty
    assert after["roe"].tolist() == [9.5]
    assert features.loc[features["trade_date"].astype(str) == "2024-01-02", "roe"].isna().all()
    assert features.loc[features["trade_date"].astype(str) == "2024-04-03", "roe"].tolist() == [
        9.5
    ]


def test_load_macro_snapshot_reads_new_raw_layout_without_legacy_fallback(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"month": "202401", "nt_val": 101.0}]),
        "raw",
        "tushare_macro_cn_cpi",
    )

    assert load_macro_snapshot(lake, as_of_date="20240131").empty

    lake.write_parquet(
        pd.DataFrame([{"month": "202401", "nt_val": 101.0}]),
        "raw",
        "tushare/cn_cpi",
    )

    frame = load_macro_snapshot(lake, as_of_date="20240131")

    assert frame[["macro_id", "period", "value"]].to_dict(orient="records") == [
        {"macro_id": "cn_cpi.nt_val", "period": "202401", "value": 101.0}
    ]
