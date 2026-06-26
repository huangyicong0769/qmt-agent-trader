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
        "tushare_daily",
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
        "tushare_daily",
        key_columns=["ts_code", "trade_date"],
    )

    merged = lake.read_parquet("raw", "tushare_daily").sort_values(["ts_code", "trade_date"])

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
        "tushare_suspend",
        key_columns=["ts_code", "trade_date"],
    )

    loaded = lake.read_parquet("raw", "tushare_suspend")
    assert list(loaded.columns) == ["ts_code", "trade_date"]
    assert loaded.empty


def test_fetch_state_and_events_are_persisted(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")

    lake.record_fetch_result(
        source="tushare",
        dataset="tushare_daily",
        start_date="20240101",
        end_date="20240103",
        status="success",
        row_count=3,
        checksum="abc",
        error=None,
    )

    state = lake.fetch_state("tushare", "tushare_daily")
    events = lake.fetch_events("tushare", "tushare_daily")

    assert state == [
        {
            "source": "tushare",
            "dataset": "tushare_daily",
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
