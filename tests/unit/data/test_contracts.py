from qmt_agent_trader.data.contracts import (
    FetchShapeName,
    find_capability_match,
    source_capability_from_field_source,
)
from qmt_agent_trader.data.field_sources import FieldSourceIndex
from qmt_agent_trader.data.providers.tushare.registry import default_tushare_registry
from qmt_agent_trader.factors.registry import FactorRegistry, input_requirements_for_factor


def test_pb_rank_requirement_matches_daily_basic_marketwide_time_slice() -> None:
    registry = default_tushare_registry()
    saved = FactorRegistry().get_factor("pb_rank")
    assert saved is not None
    requirement = input_requirements_for_factor(saved)[0]
    source = FieldSourceIndex.from_tushare_registry(registry).best_source_for_field(
        "pb",
        target_frequency=requirement.target_frequency,
    )
    assert source is not None

    capability = source_capability_from_field_source(source, registry.require(source.api_name))
    match = find_capability_match(requirement, [capability])

    assert requirement.field == "pb"
    assert requirement.target_frequency == "daily"
    assert requirement.target_calendar == "trading_days"
    assert requirement.entity_scope == "stock_cross_section"
    assert requirement.alignment_policy == "exact"
    assert match is not None
    assert match.source.source_id == "tushare.daily_basic"
    assert match.selected_fetch_shape.name == FetchShapeName.MARKETWIDE_TIME_SLICE


def test_fina_indicator_capability_exposes_asof_report_semantics() -> None:
    registry = default_tushare_registry()
    saved = FactorRegistry().get_factor("roe_rank")
    assert saved is not None
    requirement = input_requirements_for_factor(saved)[0]
    source = FieldSourceIndex.from_tushare_registry(registry).best_source_for_field(
        "roe",
        target_frequency=requirement.target_frequency,
    )
    assert source is not None

    capability = source_capability_from_field_source(source, registry.require(source.api_name))
    match = find_capability_match(requirement, [capability])

    assert requirement.alignment_policy == "asof"
    assert requirement.staleness_policy is not None
    assert capability.source_id == "tushare.fina_indicator"
    assert capability.native_frequency == "quarterly"
    assert capability.pit_model.visible_time_field == "ann_date"
    assert capability.pit_model.period_field == "end_date"
    assert match is not None
    assert match.selected_fetch_shape.name == FetchShapeName.SYMBOL_TIME_RANGE
