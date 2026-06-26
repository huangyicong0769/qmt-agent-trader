from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.factors.service import compute_factor_frame


def _bars() -> pd.DataFrame:
    start = date(2024, 1, 1)
    rows = []
    for offset in range(24):
        trade_date = start + timedelta(days=offset)
        rows.append(
            {
                "symbol": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
                "volume": 1000.0,
                "amount": 10000.0,
                "turnover": 0.01,
            }
        )
        rows.append(
            {
                "symbol": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + offset,
                "high": 21.0 + offset,
                "low": 19.0 + offset,
                "close": 20.0 + offset,
                "volume": 2000.0,
                "amount": 20000.0,
                "turnover": 0.02,
            }
        )
    return pd.DataFrame(rows)


def test_builtin_factors_are_saved_registry_entries() -> None:
    registry = FactorRegistry()

    saved = registry.get_factor("momentum_20d")

    assert saved is not None
    assert saved.factor_id == "momentum_20d"
    assert saved.status == "saved"
    assert compute_factor_frame(_bars(), "momentum_20d", registry=registry).shape[0] == 48


def test_saved_file_factor_uses_same_compute_path(tmp_path) -> None:
    factor_file = tmp_path / "factor.py"
    factor_file.write_text(
        """
from typing import Any

import pandas as pd


def compute(bars: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    lookback = int((params or {}).get("lookback", 3))
    return bars.groupby("symbol")["close"].pct_change(lookback)
""",
        encoding="utf-8",
    )
    registry = FactorRegistry(tmp_path / "registry")
    registry.save_factor(
        factor_id="agent_momentum_3d",
        name="Agent momentum 3d",
        version="0.1.0",
        implementation_ref=f"file:{factor_file}",
        required_columns=("symbol", "trade_date", "close"),
        lookback=3,
        params={"lookback": 3},
        created_by="agent",
    )

    frame = compute_factor_frame(_bars(), "agent_momentum_3d", registry=registry)

    assert frame["factor_name"].unique().tolist() == ["agent_momentum_3d"]
    assert frame["factor_value"].notna().sum() == 42


def test_unsaved_file_factor_is_not_available(tmp_path) -> None:
    factor_file = tmp_path / "draft.py"
    factor_file.write_text("def compute(bars):\n    return bars['close']\n", encoding="utf-8")
    registry = FactorRegistry(tmp_path / "registry")

    assert registry.get_factor("draft_factor") is None
