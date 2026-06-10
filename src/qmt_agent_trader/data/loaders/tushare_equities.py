"""Tushare A-share loader skeleton."""

from __future__ import annotations

from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest


def build_equity_daily_request(client: TushareClient, start: str, end: str) -> TushareRequest:
    return client.build_daily_request(start_date=start, end_date=end)
