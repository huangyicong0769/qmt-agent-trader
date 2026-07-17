from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe import resolver as resolver_module
from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.resolver import UniverseResolver


def _write_empty_trade_state_sources(lake: DataLake) -> None:
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )


def _write_stock_basic(lake: DataLake, symbols: list[str]) -> None:
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "name": symbol,
                    "list_status": "L",
                    "list_date": "20000101",
                    "delist_date": None,
                }
                for symbol in symbols
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )


def _write_calendar_sessions(lake: DataLake, session_keys: list[str]) -> None:
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": exchange,
                    "cal_date": session_key,
                    "is_open": 1,
                }
                for session_key in session_keys
                for exchange in ("SSE", "SZSE")
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )


def _write_daily_rows(
    lake: DataLake,
    rows: list[dict[str, object]],
) -> None:
    lake.write_parquet(
        pd.DataFrame(rows),
        "raw",
        "tushare/daily",
    )


def _all_stock_spec() -> UniverseSpec:
    return UniverseSpec.model_validate(
        {
            "universe_id": "all_stock",
            "name": "All stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )


def test_open_market_session_without_any_bars_fails_closed(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    _write_calendar_sessions(lake, ["20240102", "20240103"])
    _write_stock_basic(lake, ["000001.SZ"])
    _write_empty_trade_state_sources(lake)
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake).build(
            _all_stock_spec(),
            mode="snapshot",
            as_of_date="20240103",
        )

    assert exc_info.value.code == "UNIVERSE_MARKET_SESSION_NOT_READY"
    assert exc_info.value.trade_date == "2024-01-03"


def test_twenty_session_metrics_use_bounded_raw_read(tmp_path, monkeypatch) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(20)]
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"{day:%Y%m%d}",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
                for day in days
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": f"{day:%Y%m%d}",
                    "is_open": 1,
                }
                for day in days
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    observed_reads: list[dict[str, object]] = []
    original = lake.read_parquet_filtered

    def recording_read(*args, **kwargs):
        if len(args) > 1 and args[1] == "tushare/daily":
            observed_reads.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(lake, "read_parquet_filtered", recording_read)

    metrics = UniverseResolver(lake)._avg_20d_metrics(date(2024, 1, 20), ["stock"])

    assert metrics["avg_amount_20d"].tolist() == [1000.0]
    assert observed_reads == [
        {
            "start": "20240101",
            "end": "20240120",
            "columns": [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ],
        }
    ]


def test_nineteen_sessions_do_not_produce_twenty_day_liquidity(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(20)]
    _write_calendar_sessions(lake, [f"{day:%Y%m%d}" for day in days])
    _write_daily_rows(
        lake,
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": f"{day:%Y%m%d}",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "vol": 100.0,
                "amount": 1000.0,
            }
            for day in days[1:]
        ],
    )

    observed = UniverseResolver(lake)._avg_20d_metrics(
        days[-1],
        ["stock"],
    )

    assert observed.loc[0, "amount_observation_count"] == 19
    assert observed.loc[0, "volume_observation_count"] == 19
    assert pd.isna(observed.loc[0, "avg_amount_20d"])
    assert pd.isna(observed.loc[0, "avg_volume_20d"])


def test_null_amount_invalidates_only_amount_window(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(20)]
    _write_calendar_sessions(lake, [f"{day:%Y%m%d}" for day in days])
    rows = []
    for index, day in enumerate(days):
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": f"{day:%Y%m%d}",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "vol": 100.0,
                "amount": None if index == 0 else 1000.0,
            }
        )
    _write_daily_rows(lake, rows)

    observed = UniverseResolver(lake)._avg_20d_metrics(
        days[-1],
        ["stock"],
    )

    assert observed.loc[0, "amount_observation_count"] == 19
    assert observed.loc[0, "volume_observation_count"] == 20
    assert pd.isna(observed.loc[0, "avg_amount_20d"])
    assert observed.loc[0, "avg_volume_20d"] == 100.0


def test_closed_boundary_uses_previous_open_session_for_all_pit_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SSE", "cal_date": "20240107", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240107", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    observed: dict[str, object] = {}
    resolver = UniverseResolver(lake)
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "closed-boundary",
            "name": "Closed boundary",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "index_constituents",
                "index_codes": ["000300.SH"],
            },
            "filters": {"min_listed_days": 0},
        }
    )

    def fake_bars(effective_date, _asset_types):
        observed["bars_date"] = effective_date
        return pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": effective_date,
                    "asset_type": "stock",
                    "st": False,
                    "suspended": False,
                    "volume": 100.0,
                    "amount": 1000.0,
                }
            ]
        )

    def fake_index(_codes, effective_date):
        observed["index_date"] = effective_date
        return ["000001.SZ"]

    def fake_metrics(frame, _spec, *, effective_date):
        observed["metrics_date"] = effective_date
        return frame

    monkeypatch.setattr(resolver, "_load_recent_bars", fake_bars)
    monkeypatch.setattr(
        resolver,
        "_stock_basic",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "Fixture",
                    "list_date": "20000101",
                    "delist_date": None,
                }
            ]
        ),
    )
    monkeypatch.setattr(resolver, "_index_constituents", fake_index)
    monkeypatch.setattr(resolver, "_attach_metrics", fake_metrics)

    result = resolver.build(
        spec,
        mode="snapshot",
        as_of_date="20240107",
    )

    assert result["status"] == "OK"
    assert result["symbols"] == ["000001.SZ"]
    assert observed == {
        "bars_date": date(2024, 1, 5),
        "index_date": date(2024, 1, 5),
        "metrics_date": date(2024, 1, 5),
    }
    diagnostics = result["metadata"]["diagnostics"]
    assert diagnostics["requested_as_of_date"] == "20240107"
    assert diagnostics["effective_market_session"] == "20240105"


def test_mixed_universe_requires_each_requested_asset_type(
    tmp_path,
    monkeypatch,
) -> None:
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
    monkeypatch.setattr(
        resolver_module,
        "load_daily_bars",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": date(2024, 1, 2),
                    "asset_type": "stock",
                }
            ]
        ),
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        resolver._load_recent_bars(
            date(2024, 1, 2),
            ["stock", "etf"],
        )

    assert exc_info.value.code == "UNIVERSE_MARKET_SESSION_NOT_READY"
    assert exc_info.value.details == {
        "requested_asset_types": ["etf", "stock"],
        "observed_asset_types": ["stock"],
        "missing_asset_types": ["etf"],
    }


def test_mixed_universe_accepts_stock_and_etf_rows(
    tmp_path,
    monkeypatch,
) -> None:
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
    expected = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "asset_type": "stock",
            },
            {
                "symbol": "510300.SH",
                "trade_date": date(2024, 1, 2),
                "asset_type": "etf",
            },
        ]
    )
    monkeypatch.setattr(
        resolver_module,
        "load_daily_bars",
        lambda *_args, **_kwargs: expected.copy(),
    )

    observed = resolver._load_recent_bars(
        date(2024, 1, 2),
        ["stock", "etf"],
    )

    assert observed["asset_type"].tolist() == ["stock", "etf"]


def test_etf_category_without_dated_classification_fails_closed(
    tmp_path,
) -> None:
    lake = DataLake(
        tmp_path / "lake",
        tmp_path / "research.duckdb",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": "20240102",
                    "is_open": 1,
                },
                {
                    "exchange": "SZSE",
                    "cal_date": "20240102",
                    "is_open": 1,
                },
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20240102",
                    "open": 3.5,
                    "high": 3.6,
                    "low": 3.4,
                    "close": 3.55,
                    "vol": 100.0,
                    "amount": 350.0,
                }
            ]
        ),
        "raw",
        "tushare/fund_daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20240102",
                    "up_limit": 3.85,
                    "down_limit": 3.15,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "etf-category",
            "name": "ETF category",
            "source": "user_defined",
            "asset_types": ["etf"],
            "selection": {
                "mode": "etf_category",
                "theme_concepts": ["broad_market"],
            },
            "filters": {"min_listed_days": 0},
        }
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake).build(
            spec,
            as_of_date="20240102",
        )

    assert exc_info.value.code == "UNIVERSE_PIT_CLASSIFICATION_NOT_READY"
    assert exc_info.value.field == "classification_history"
    assert exc_info.value.details["selection_mode"] == "etf_category"


def test_etf_category_candidate_helper_never_falls_back_to_all_etfs(
    tmp_path,
) -> None:
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
    recent = pd.DataFrame(
        [
            {
                "symbol": "510300.SH",
                "asset_type": "etf",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        resolver._etf_category_candidates(
            ["broad_market"],
            recent,
        )

    assert exc_info.value.code == "UNIVERSE_PIT_CLASSIFICATION_NOT_READY"
