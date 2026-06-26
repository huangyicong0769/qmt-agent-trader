from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.query_tools import list_data_catalog_tool, set_data_lake
from qmt_agent_trader.data.storage import DataLake


def test_list_data_catalog_hides_legacy_tushare_batches(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    frame = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102"}])
    lake.write_parquet(frame, "raw", "tushare_daily")
    lake.write_parquet(frame, "raw", "tushare_daily_adjusted")
    lake.write_parquet(frame, "raw", "tushare_daily_20240101_20240103")
    lake.write_parquet(frame, "raw", "tushare_suspend_20240101_20240103")
    lake.write_parquet(frame, "raw", "tushare_stk_limit_20240101_20240103")
    lake.write_parquet(frame, "gold", "factor_momentum_20d_20240102")
    set_data_lake(lake)

    result = list_data_catalog_tool.run({}, ToolContext(run_id="catalog"))

    assert result["status"] == "ok"
    assert "tushare_daily" in result["layers"]["raw"]
    assert "tushare_daily_adjusted" in result["layers"]["raw"]
    assert "factor_momentum_20d_20240102" in result["layers"]["gold"]
    assert "tushare_daily_20240101_20240103" not in result["layers"]["raw"]
    assert "tushare_suspend_20240101_20240103" not in result["layers"]["raw"]
    assert "tushare_stk_limit_20240101_20240103" not in result["layers"]["raw"]
