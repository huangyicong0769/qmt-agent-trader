"""Tushare trading calendar loader skeleton."""

from __future__ import annotations

from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest


def build_calendar_request(client: TushareClient, start: str, end: str) -> TushareRequest:
    return client.build_trade_calendar_request(start_date=start, end_date=end)
