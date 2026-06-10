"""QMT market data adapter backed by the remote gateway."""

from __future__ import annotations

from qmt_agent_trader.broker.remote_client import RemoteQMTBrokerClient


class QMTMarketClient:
    def __init__(self, broker_client: RemoteQMTBrokerClient) -> None:
        self.broker_client = broker_client

    def latest(self, symbols: list[str]) -> dict[str, object]:
        return self.broker_client.get_latest_quotes(symbols)

    def bars(self, symbols: list[str], start: str, end: str, freq: str) -> dict[str, object]:
        return self.broker_client.get_bars(symbols=symbols, start=start, end=end, freq=freq)
