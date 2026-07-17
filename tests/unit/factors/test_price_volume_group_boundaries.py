from __future__ import annotations

import numpy as np
import pandas as pd

from qmt_agent_trader.factors.library.price_volume import volatility_20d


def test_volatility_window_never_crosses_symbol_boundary() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    frame = pd.DataFrame(
        [
            row
            for index, day in enumerate(dates)
            for row in (
                {
                    "symbol": "A",
                    "trade_date": day,
                    "close": 100.0 + float(index),
                },
                {
                    "symbol": "B",
                    "trade_date": day,
                    "close": 10.0 * (2.0**index),
                },
            )
        ]
    )

    observed = volatility_20d(frame)
    expected = (
        frame.groupby("symbol", sort=False)["close"]
        .pct_change()
        .groupby(frame["symbol"], sort=False)
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
        .reindex(frame.index)
    )

    pd.testing.assert_series_equal(observed, expected, check_names=False)
    assert np.isnan(observed.iloc[:38]).all()
