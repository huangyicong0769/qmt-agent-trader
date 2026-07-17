from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import (
    BacktestDataIntegrityError,
    BacktestUniverseIntegrityError,
)
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseRule, UniverseSpec
from qmt_agent_trader.universe.resolver import (
    UniverseResolver,
    _apply_limit,
    _ordered_unique_symbols,
    _period_end_dates,
)


def _spec(max_symbols=None) -> UniverseSpec:
    return UniverseSpec.model_validate(
        {
            "universe_id": "all_stock",
            "name": "All stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "max_symbols": max_symbols,
        }
    )


def test_broad_universe_is_not_implicitly_cut_to_2000() -> None:
    selected, metadata = _apply_limit(
        [str(index) for index in range(3_000)], spec=_spec(), limit=None
    )
    assert len(selected) == 3_000
    assert metadata["truncated"] is False


def test_explicit_limit_is_reported_as_truncation() -> None:
    selected, metadata = _apply_limit(
        [str(index) for index in range(3_000)],
        spec=_spec(),
        limit=2_000,
    )
    assert len(selected) == 2_000
    assert metadata == {
        "pre_limit_selected_count": 3_000,
        "selected_count": 2_000,
        "truncated": True,
        "effective_limit": 2_000,
        "truncation_source": "request_limit",
    }


def test_weekly_resolve_dates_use_anchor_and_period_ends() -> None:
    dates = ["20240102", "20240103", "20240105", "20240108", "20240112"]

    assert _period_end_dates(dates, "weekly") == ["20240102", "20240105", "20240112"]


def test_monthly_resolve_dates_use_anchor_and_period_ends() -> None:
    dates = ["20240102", "20240131", "20240201", "20240229"]

    assert _period_end_dates(dates, "monthly") == ["20240102", "20240131", "20240229"]


def test_ranked_universe_limit_preserves_ranking_order() -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked",
            "name": "Ranked",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {"field": "avg_amount_20d", "ascending": False},
            "max_symbols": 2,
        }
    )
    frame = pd.DataFrame(
        {
            "symbol": ["000003.SZ", "000001.SZ", "000002.SZ"],
            "avg_amount_20d": [300.0, 200.0, 100.0],
        }
    )

    symbols = _ordered_unique_symbols(frame, spec)
    selected, _ = _apply_limit(symbols, spec=spec, limit=None)

    assert selected == ["000003.SZ", "000001.SZ"]


def test_ranked_universe_ties_use_symbol_ascending_tiebreak(tmp_path) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked",
            "name": "Ranked",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {"field": "avg_amount_20d", "ascending": False},
            "max_symbols": 2,
        }
    )
    frame = pd.DataFrame(
        {
            "symbol": ["000003.SZ", "000001.SZ", "000002.SZ"],
            "avg_amount_20d": [100.0, 100.0, 100.0],
            "amount_observation_count": [20, 20, 20],
        }
    )
    resolver = UniverseResolver(DataLake(tmp_path / "lake", tmp_path / "research.duckdb"))

    ranked = resolver._apply_ranking(frame, spec)
    symbols = _ordered_unique_symbols(ranked, spec)
    selected, _ = _apply_limit(symbols, spec=spec, limit=None)

    assert selected == ["000001.SZ", "000002.SZ"]


def test_explicit_symbol_order_is_preserved() -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "explicit",
            "name": "Explicit",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "explicit_symbols",
                "symbols": ["000003.SZ", "000001.SZ", "000002.SZ"],
            },
        }
    )
    frame = pd.DataFrame({"symbol": spec.selection.symbols})
    assert _ordered_unique_symbols(frame, spec) == spec.selection.symbols


def test_amount_filter_rejects_incomplete_twenty_day_window() -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "amount_filter",
            "name": "Amount filter",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {
                "min_listed_days": 0,
                "min_avg_amount_20d": 1.0,
            },
        }
    )
    row = {
        "symbol": "000001.SZ",
        "asset_type": "stock",
        "has_bar_coverage": True,
        "listed_as_of": True,
        "list_date": "20000101",
        "st": False,
        "suspended": False,
        "avg_amount_20d": pd.NA,
        "amount_observation_count": 19,
    }

    reason = UniverseResolver.__new__(UniverseResolver)._exclusion_reason(
        spec,
        row,
        as_of_date="20240131",
    )

    assert reason == "amount_20d_coverage_incomplete"


def test_multiple_indices_resolve_from_independent_sources(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000300.SH",
                    "con_code": "000001.SZ",
                    "trade_date": "20240201",
                }
            ]
        ),
        "raw",
        "tushare/index_weight",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000905.SH",
                    "con_code": "000002.SZ",
                    "in_date": "20240101",
                    "out_date": None,
                }
            ]
        ),
        "raw",
        "tushare/index_member",
    )

    observed = UniverseResolver(lake)._index_constituents(
        ["000300.SH", "000905.SH"],
        date(2024, 2, 15),
    )

    assert observed == ["000001.SZ", "000002.SZ"]


def test_missing_one_requested_index_fails_closed(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000300.SH",
                    "con_code": "000001.SZ",
                    "trade_date": "20240201",
                }
            ]
        ),
        "raw",
        "tushare/index_weight",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake)._index_constituents(
            ["000300.SH", "000905.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_NOT_READY"
    assert exc_info.value.details["missing_index_codes"] == ["000905.SH"]


def test_resolver_preserves_ranked_top_symbols_before_limit(tmp_path, monkeypatch) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked",
            "name": "Ranked",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {"field": "avg_amount_20d", "ascending": False},
            "max_symbols": 2,
        }
    )
    candidates = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "avg_amount_20d": [200.0, 100.0, 300.0],
            "amount_observation_count": [20, 20, 20],
        }
    )
    resolver = UniverseResolver(DataLake(tmp_path / "lake", tmp_path / "research.duckdb"))
    monkeypatch.setattr(resolver, "_load_recent_bars", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(resolver, "_stock_basic", lambda: pd.DataFrame())
    monkeypatch.setattr(
        resolver,
        "_select_candidates",
        lambda *_args, **_kwargs: candidates.copy(),
    )
    monkeypatch.setattr(
        resolver,
        "_attach_metrics",
        lambda frame, *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        resolver,
        "_exclusion_reason",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "qmt_agent_trader.universe.resolver.latest_open_session_on_or_before",
        lambda *_args, **_kwargs: pd.Timestamp("2024-01-31").date(),
    )

    symbols, _, _ = resolver._resolve_for_date(spec, as_of_date="20240131")
    selected, _ = _apply_limit(symbols, spec=spec, limit=None)

    assert selected == ["000003.SZ", "000001.SZ"]


def test_snapshot_uses_validated_non_null_trade_state(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1_000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )
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
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "Example Co",
                    "list_status": "L",
                    "list_date": "20200101",
                }
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    lake.write_parquet(
        pd.DataFrame(
            [{"exchange": "SSE", "cal_date": "20240102", "is_open": 1}]
        ),
        "raw",
        "tushare/trade_cal",
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "validated-stock",
            "name": "Validated stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
        }
    )

    result = UniverseResolver(lake).build(spec, as_of_date="20240102")

    assert result["status"] == "OK"
    assert result["symbols"] == ["000001.SZ"]


def test_snapshot_selects_etf_with_exact_session_bar(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
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
    lake.write_parquet(
        pd.DataFrame(
            [{"exchange": "SSE", "cal_date": "20240102", "is_open": 1}]
        ),
        "raw",
        "tushare/trade_cal",
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "validated-etf",
            "name": "Validated ETF",
            "source": "user_defined",
            "asset_types": ["etf"],
            "selection": {"mode": "all"},
        }
    )

    result = UniverseResolver(lake).build(spec, as_of_date="20240102")

    assert result["status"] == "OK"
    assert result["symbols"] == ["510300.SH"]


def test_market_cap_asof_rejects_duplicate_symbol_date(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "total_mv": 100.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "total_mv": 101.0,
                },
            ]
        ),
        "raw",
        "tushare/daily_basic",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        UniverseResolver(lake)._market_cap_asof(date(2024, 1, 2))

    assert exc_info.value.code == "DUPLICATE_UNIVERSE_SOURCE_KEY"


def test_liquidity_ranking_does_not_fill_top_n_with_incomplete_window(
    tmp_path,
) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked-liquidity",
            "name": "Ranked liquidity",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {
                "field": "avg_amount_20d",
                "ascending": False,
                "top_n": 2,
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_amount_20d": 1000.0,
                "amount_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_amount_20d": pd.NA,
                "amount_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    ranked = resolver._apply_ranking(frame, spec)

    assert ranked["symbol"].tolist() == ["000001.SZ"]


def test_volume_ranking_requires_twenty_non_null_observations(
    tmp_path,
) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked-volume",
            "name": "Ranked volume",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {
                "field": "avg_volume_20d",
                "ascending": False,
                "top_n": 5,
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_volume_20d": 100.0,
                "volume_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_volume_20d": 200.0,
                "volume_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    ranked = resolver._apply_ranking(frame, spec)

    assert ranked["symbol"].tolist() == ["000001.SZ"]


def test_expired_index_history_is_not_valid_asof_evidence(
    tmp_path,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000905.SH",
                    "con_code": "000001.SZ",
                    "in_date": "20200101",
                    "out_date": "20231231",
                }
            ]
        ),
        "raw",
        "tushare/index_member",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake)._index_constituents(
            ["000905.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_NOT_READY"
    assert exc_info.value.details["missing_index_codes"] == ["000905.SH"]


def test_liquidity_rule_ne_rejects_incomplete_window(tmp_path) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "amount-rule-ne",
            "name": "Amount rule ne",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "all",
                "rules": [
                    {
                        "field": "avg_amount_20d",
                        "operator": "ne",
                        "value": 0,
                    }
                ],
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_amount_20d": 1000.0,
                "amount_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_amount_20d": pd.NA,
                "amount_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, spec.selection.rules)

    assert observed["symbol"].tolist() == ["000001.SZ"]


def test_liquidity_rule_not_in_rejects_missing_evidence(tmp_path) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "volume-rule-not-in",
            "name": "Volume rule not in",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "all",
                "rules": [
                    {
                        "field": "avg_volume_20d",
                        "operator": "not_in",
                        "value": [0],
                    }
                ],
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_volume_20d": 100.0,
                "volume_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_volume_20d": pd.NA,
                "volume_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, spec.selection.rules)

    assert observed["symbol"].tolist() == ["000001.SZ"]


def test_liquidity_rule_requires_observation_count_column(tmp_path) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "amount-rule-count",
            "name": "Amount rule count",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "all",
                "rules": [
                    {
                        "field": "avg_amount_20d",
                        "operator": "gt",
                        "value": 10,
                    }
                ],
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_amount_20d": 1000.0,
            }
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, spec.selection.rules)

    assert observed.empty


def test_liquidity_rule_rejects_non_finite_metric(tmp_path) -> None:
    rule = UniverseRule.model_validate(
        {
            "field": "avg_amount_20d",
            "operator": "gt",
            "value": 10,
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_amount_20d": 100.0,
                "amount_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_amount_20d": float("inf"),
                "amount_observation_count": 20,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, [rule])

    assert observed["symbol"].tolist() == ["000001.SZ"]


def test_non_liquidity_rule_keeps_existing_semantics(tmp_path) -> None:
    rule = UniverseRule.model_validate(
        {
            "field": "market_cap",
            "operator": "gte",
            "value": 100,
        }
    )
    frame = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "market_cap": 100.0},
            {"symbol": "000002.SZ", "market_cap": 99.0},
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, [rule])

    assert observed["symbol"].tolist() == ["000001.SZ"]
