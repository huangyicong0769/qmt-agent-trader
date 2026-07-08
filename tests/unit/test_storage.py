import pandas as pd

from qmt_agent_trader.data.storage import DataLake


def test_duckdb_parquet_roundtrip(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    frame = pd.DataFrame({"symbol": ["000001.SZ"], "close": [10.0]})
    lake.write_parquet(frame, "raw", "bars")
    loaded = lake.read_parquet("raw", "bars")
    assert loaded.to_dict("records") == [{"symbol": "000001.SZ", "close": 10.0}]
    lake.register_parquet("bars", "raw", "bars")
    queried = lake.query_parquet("select symbol, close from bars")
    assert queried.iloc[0]["symbol"] == "000001.SZ"


def test_list_dataset_names_includes_nested_registry_dataset_ids(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(pd.DataFrame([{"ts_code": "000001.SZ"}]), "raw", "tushare/daily")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ"}]),
        "raw",
        "tushare/daily_basic",
    )

    assert lake.list_dataset_names("raw", prefix="tushare/") == [
        "tushare/daily",
        "tushare/daily_basic",
    ]


def test_incremental_parquet_merges_by_key(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")

    lake.write_incremental_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.0},
                {"ts_code": "000002.SZ", "trade_date": "20240102", "close": 20.0},
            ]
        ),
        "raw",
        "tushare/daily",
        key_columns=["ts_code", "trade_date"],
    )
    lake.write_incremental_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.5},
                {"ts_code": "000003.SZ", "trade_date": "20240103", "close": 30.0},
            ]
        ),
        "raw",
        "tushare/daily",
        key_columns=["ts_code", "trade_date"],
    )

    merged = lake.read_parquet("raw", "tushare/daily").sort_values(["ts_code", "trade_date"])

    assert merged.to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.5},
        {"ts_code": "000002.SZ", "trade_date": "20240102", "close": 20.0},
        {"ts_code": "000003.SZ", "trade_date": "20240103", "close": 30.0},
    ]


def test_incremental_parquet_accepts_empty_fetch_frames(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")

    lake.write_incremental_parquet(
        pd.DataFrame(),
        "raw",
        "tushare/suspend_d",
        key_columns=["ts_code", "trade_date"],
    )

    loaded = lake.read_parquet("raw", "tushare/suspend_d")
    assert list(loaded.columns) == ["ts_code", "trade_date"]
    assert loaded.empty


def test_catalog_exposes_unmigrated_legacy_batches_instead_of_hiding_them(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 10.0},
                {"ts_code": "000002.SZ", "trade_date": "20240102", "close": 20.0},
            ]
        ),
        "raw",
        "tushare_daily_20240101_20240102",
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

    assert lake.list_dataset_names("raw", prefix="tushare_daily_") == [
        "tushare_daily_20240101_20240102",
        "tushare_daily_20240102_20240103",
        "tushare_daily_adjusted",
    ]
    assert lake.dataset_path("raw", "tushare_daily_adjusted").exists()
    assert lake.dataset_path("raw", "tushare_daily_20240101_20240102").exists()
    assert lake.dataset_path("raw", "tushare_daily_20240102_20240103").exists()
    assert not lake.dataset_path("raw", "tushare/daily").exists()


def test_fetch_state_and_events_are_persisted(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")

    lake.record_fetch_result(
        source="tushare",
        dataset="tushare.daily",
        start_date="20240101",
        end_date="20240103",
        status="success",
        row_count=3,
        checksum="abc",
        error=None,
    )

    state = lake.fetch_state("tushare", "tushare.daily")
    events = lake.fetch_events("tushare", "tushare.daily")

    assert state == [
        {
            "source": "tushare",
            "dataset": "tushare.daily",
            "start_date": "20240101",
            "end_date": "20240103",
            "status": "success",
            "row_count": 3,
            "checksum": "abc",
            "error": None,
        }
    ]
    assert len(events) == 1
    assert events[0]["status"] == "success"


def test_read_parquet_filtered_returns_empty_for_missing_dataset(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")

    loaded = lake.read_parquet_filtered("raw", "missing", start="20240101")

    assert loaded.empty


def test_read_parquet_filtered_pushes_date_symbol_and_column_filters(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": 20240102, "close": 10.0, "open": 9.5},
                {"ts_code": "000002.SZ", "trade_date": 20240103, "close": 20.0, "open": 19.5},
                {"ts_code": "000001.SZ", "trade_date": 20240104, "close": 11.0, "open": 10.5},
            ]
        ),
        "raw",
        "tushare/daily",
    )

    loaded = lake.read_parquet_filtered(
        "raw",
        "tushare/daily",
        columns=["ts_code", "trade_date", "close"],
        start="20240103",
        end="20240104",
        symbols=["000001.SZ"],
    )

    assert list(loaded.columns) == ["ts_code", "trade_date", "close"]
    assert loaded.to_dict("records") == [
        {"ts_code": "000001.SZ", "trade_date": 20240104, "close": 11.0}
    ]


def test_read_parquet_filtered_handles_string_and_date_like_trade_dates(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": pd.Timestamp("2024-01-02").date(),
                    "close": 10.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": pd.Timestamp("2024-01-03").date(),
                    "close": 11.0,
                },
            ]
        ),
        "raw",
        "date_dates",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 20.0},
                {"ts_code": "000001.SZ", "trade_date": "20240103", "close": 21.0},
            ]
        ),
        "raw",
        "string_dates",
    )

    loaded_dates = lake.read_parquet_filtered("raw", "date_dates", start="20240103")
    loaded_strings = lake.read_parquet_filtered("raw", "string_dates", start="20240103")

    assert loaded_dates["close"].tolist() == [11.0]
    assert loaded_strings["close"].tolist() == [21.0]
