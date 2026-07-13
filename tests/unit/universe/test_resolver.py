from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.resolver import _apply_limit


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
