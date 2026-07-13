from datetime import date

from qmt_agent_trader.backtest.rebalance import (
    build_execution_schedule,
    select_signal_dates,
)


def test_weekly_schedule_uses_last_trading_day_of_iso_week() -> None:
    dates = tuple(date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9, 10, 11, 12))
    assert select_signal_dates(dates, "weekly") == (date(2024, 1, 5), date(2024, 1, 12))


def test_execution_schedule_applies_delay_in_trading_sessions() -> None:
    dates = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    assert build_execution_schedule(
        dates,
        signal_dates=(date(2024, 1, 2),),
        delay_days=1,
    ) == {date(2024, 1, 3): date(2024, 1, 2)}
