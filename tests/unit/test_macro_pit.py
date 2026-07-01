from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.transforms.macro_pit import load_macro_series_asof


def test_load_macro_series_asof_filters_by_visible_date(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"date": "20240102", "on": 1.5},
                {"date": "20240201", "on": 1.6},
            ]
        ),
        "raw",
        "tushare_macro_shibor",
    )

    frame, metadata = load_macro_series_asof(
        lake,
        dataset="shibor",
        as_of_date="20240131",
        fields=["on"],
    )

    assert metadata["status"] == "OK"
    assert metadata["pit_safe"] is True
    assert frame["date"].tolist() == ["20240102"]
    assert frame["visible_date"].astype(str).tolist() == ["2024-01-02"]
