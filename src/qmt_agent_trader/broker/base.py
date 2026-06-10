"""Broker interface."""

from __future__ import annotations

from typing import Protocol


class BrokerClient(Protocol):
    def health(self) -> dict[str, object]: ...

    def query_asset(self) -> dict[str, object]: ...

    def query_positions(self) -> dict[str, object]: ...
