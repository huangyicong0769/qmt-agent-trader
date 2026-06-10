from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.service import compute_factor_to_lake, validate_factor


def test_compute_factor_to_lake(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    start = date(2024, 1, 1)
    rows = [
        {
            "ts_code": "000001.SZ",
            "trade_date": f"{start + timedelta(days=offset):%Y%m%d}",
            "open": 10.0 + offset,
            "high": 11.0 + offset,
            "low": 9.0 + offset,
            "close": 10.0 + offset,
            "vol": 1000.0,
            "amount": 10000.0,
        }
        for offset in range(21)
    ]
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily_fixture")

    result = compute_factor_to_lake(lake, name="momentum_20d", date="20240121")

    assert result.rows == 1
    assert result.non_null == 1
    assert lake.dataset_path("gold", "factor_momentum_20d_20240121").exists()


def test_validate_factor_computes_ic(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    start = date(2024, 1, 1)
    rows = []
    for offset in range(22):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
            }
        )
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + offset,
                "high": 21.0 + offset,
                "low": 19.0 + offset,
                "close": 20.0 + offset,
            }
        )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily_fixture")

    result = validate_factor(lake, name="momentum_20d", start="20240121", end="20240121")

    assert result.observations == 2
    assert result.non_null == 2
