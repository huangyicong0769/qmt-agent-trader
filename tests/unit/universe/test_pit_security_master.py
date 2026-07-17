from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.pit_metadata import security_master_asof
from qmt_agent_trader.universe.resolver import UniverseResolver


def test_delisted_symbol_is_eligible_before_delist_date() -> None:
    current = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "Current Name",
                "list_status": "D",
                "list_date": "20000101",
                "delist_date": "20200110",
            }
        ]
    )

    observed = security_master_asof(current, date(2020, 1, 5))

    assert observed["symbol"].tolist() == ["000001.SZ"]
    assert observed["listed_as_of"].tolist() == [True]


def test_future_st_name_is_not_historical_st_evidence() -> None:
    current = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "ST Current Name",
                "list_status": "L",
                "list_date": "20000101",
                "delist_date": None,
            }
        ]
    )

    observed = security_master_asof(current, date(2010, 1, 5))

    assert observed["display_name"].tolist() == ["ST Current Name"]
    assert "st" not in observed.columns


def test_resolver_ignores_current_status_and_name_for_historical_date(
    tmp_path,
    monkeypatch,
) -> None:
    resolver = UniverseResolver(DataLake(tmp_path / "lake", tmp_path / "research.duckdb"))
    monkeypatch.setattr(
        resolver,
        "_load_recent_bars",
        lambda *_args: pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": date(2020, 1, 5),
                    "asset_type": "stock",
                    "st": False,
                    "suspended": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        resolver,
        "_stock_basic",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "ST Current Name",
                    "list_status": "D",
                    "list_date": "20000101",
                    "delist_date": "20200110",
                }
            ]
        ),
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "historical-stock",
            "name": "Historical stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )

    symbols, excluded, _ = resolver._resolve_for_date(
        spec,
        as_of_date="20200105",
    )

    assert symbols == ["000001.SZ"]
    assert excluded == []


def test_historical_industry_selection_requires_dated_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    resolver = UniverseResolver(DataLake(tmp_path / "lake", tmp_path / "research.duckdb"))
    monkeypatch.setattr(resolver, "_load_recent_bars", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(resolver, "_stock_basic", lambda: pd.DataFrame())
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "historical-industry",
            "name": "Historical industry",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "industry", "industries": ["银行"]},
        }
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        resolver._resolve_for_date(spec, as_of_date="20200105")

    assert exc_info.value.code == "UNIVERSE_PIT_CLASSIFICATION_NOT_READY"
