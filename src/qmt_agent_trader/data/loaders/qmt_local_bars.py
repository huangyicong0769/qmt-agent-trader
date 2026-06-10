"""QMT local bar sync skeleton."""

from __future__ import annotations

from qmt_agent_trader.data.qmt_market_client import QMTMarketClient


def load_qmt_bars(
    client: QMTMarketClient, symbols: list[str], start: str, end: str, freq: str
) -> dict[str, object]:
    return client.bars(symbols=symbols, start=start, end=end, freq=freq)
