from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from qmt_agent_trader.data.storage import DataLake


def _module():
    path = Path(__file__).parents[2] / "scripts/profile_research_tools.py"
    spec = importlib.util.spec_from_file_location("profile_research_tools", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_helpers_query_external_parquet(tmp_path: Path) -> None:
    module = _module()
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "catalog.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102"},
                {"ts_code": "000002.SZ", "trade_date": "20240103"},
            ]
        ),
        "raw",
        "tushare_daily",
    )

    assert module._date_bounds(lake) == {"start": "20240102", "end": "20240103"}
    assert module._sample_symbols(
        lake, start="20240102", end="20240103", limit=10
    ) == ["000001.SZ", "000002.SZ"]
