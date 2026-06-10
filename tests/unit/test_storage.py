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
