from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.resolver import _apply_limit, _period_end_dates


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
