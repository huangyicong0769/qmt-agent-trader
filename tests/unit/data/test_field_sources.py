from __future__ import annotations

from qmt_agent_trader.data.field_sources import FieldSourceIndex, FillPolicy
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.providers.tushare.registry import default_tushare_registry


def test_daily_basic_field_resolves_to_exact_daily_source() -> None:
    index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())

    source = index.best_source_for_field("dv_ttm", target_frequency=Frequency.DAILY)

    assert source is not None
    assert source.api_name == "daily_basic"
    assert source.raw_dataset_name == "tushare/daily_basic"
    assert source.entity_column == "ts_code"
    assert source.canonical_entity_column == "symbol"
    assert source.source_time_column == "trade_date"
    assert source.visible_time_column == "trade_date"
    assert source.frequency is Frequency.DAILY
    assert source.fill_policy is FillPolicy.EXACT
    assert source.pit_safe is True


def test_financial_field_resolves_to_low_frequency_pit_source() -> None:
    index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())

    source = index.best_source_for_field("debt_to_assets", target_frequency=Frequency.DAILY)

    assert source is not None
    assert source.api_name == "fina_indicator"
    assert source.raw_dataset_name == "tushare/fina_indicator"
    assert source.source_time_column == "end_date"
    assert source.visible_time_column == "ann_date"
    assert source.frequency is Frequency.QUARTERLY
    assert source.fill_policy is FillPolicy.ASOF_SNAPSHOT
    assert source.pit_safe is True


def test_monthly_macro_field_resolves_to_pit_asof_source() -> None:
    index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())

    source = index.best_source_for_field("nt_val", target_frequency=Frequency.DAILY)

    assert source is not None
    assert source.api_name == "cn_cpi"
    assert source.entity_column is None
    assert source.source_time_column == "month"
    assert source.visible_time_column == "visible_date"
    assert source.frequency is Frequency.MONTHLY
    assert source.fill_policy is FillPolicy.ASOF_SNAPSHOT
    assert source.pit_safe is True


def test_corporate_action_event_field_does_not_default_to_asof_fill() -> None:
    index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())

    source = index.best_source_for_field("cash_div", target_frequency=Frequency.DAILY)

    assert source is not None
    assert source.api_name == "dividend"
    assert source.frequency is Frequency.EVENT
    assert source.fill_policy is FillPolicy.NO_FILL


def test_unknown_field_has_no_source() -> None:
    index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())

    assert index.sources_for_field("mystery_ratio") == []
    assert index.best_source_for_field("mystery_ratio", target_frequency=Frequency.DAILY) is None


def test_ambiguous_multi_source_field_is_not_silently_selected() -> None:
    index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())

    candidates = index.sources_for_field("close")

    assert {candidate.api_name for candidate in candidates} >= {
        "daily",
        "daily_basic",
        "fund_daily",
    }
    assert index.best_source_for_field("close", target_frequency=Frequency.DAILY) is None
