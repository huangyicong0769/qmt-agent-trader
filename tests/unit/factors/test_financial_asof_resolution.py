from __future__ import annotations

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.fundamentals import financial_field_asof_source


def test_same_day_financial_revision_uses_latest_period_and_update() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20230930",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "0",
                "roe": 8.0,
            },
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "1",
                "roe": 10.0,
            },
        ]
    )

    observed = financial_field_asof_source(
        raw,
        field="roe",
        source="tushare/fina_indicator",
    )

    assert observed.to_dict(orient="records") == [
        {
            "symbol": "000001.SZ",
            "visible_date": pd.Timestamp("2024-04-30").date(),
            "roe": 10.0,
        }
    ]


def test_identical_business_rank_with_conflicting_value_fails() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "1",
                "roe": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "1",
                "roe": 11.0,
            },
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        financial_field_asof_source(
            raw,
            field="roe",
            source="tushare/fina_indicator",
        )

    assert exc_info.value.code == "AMBIGUOUS_FINANCIAL_REVISION"
