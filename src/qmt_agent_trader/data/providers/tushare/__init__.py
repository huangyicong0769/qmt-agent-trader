"""Registry-driven Tushare provider."""

from qmt_agent_trader.data.providers.tushare.client import TushareClient
from qmt_agent_trader.data.providers.tushare.provider import TushareProvider
from qmt_agent_trader.data.providers.tushare.registry import (
    EndpointSpec,
    TushareEndpointRegistry,
    default_tushare_registry,
)

__all__ = [
    "EndpointSpec",
    "TushareClient",
    "TushareEndpointRegistry",
    "TushareProvider",
    "default_tushare_registry",
]
