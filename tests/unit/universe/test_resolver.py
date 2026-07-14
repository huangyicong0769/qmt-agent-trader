import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseSpec
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
        }
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

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
        }
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
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

    symbols, _, _ = resolver._resolve_for_date(spec, as_of_date="20240131")
    selected, _ = _apply_limit(symbols, spec=spec, limit=None)

    assert selected == ["000003.SZ", "000001.SZ"]
