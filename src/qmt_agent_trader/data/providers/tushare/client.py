"""Thin generic Tushare query wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd

from qmt_agent_trader.core.errors import ConfigurationError


class TushareClient:
    """Generic client for registry/planner driven Tushare calls."""

    def __init__(self, token: str | None, *, timeout_seconds: float = 300.0) -> None:
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._pro: Any | None = None

    def query(
        self,
        api_name: str,
        params: dict[str, Any],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        api = self.pro()
        kwargs = dict(params)
        if fields is not None:
            kwargs["fields"] = ",".join(fields)
        return api.query(api_name, **kwargs)

    def pro(self) -> Any:
        if not self.token:
            raise ConfigurationError("TUSHARE_TOKEN is required for live Tushare requests")
        if self._pro is None:
            import tushare as ts

            ts.set_token(self.token)
            self._pro = ts.pro_api(self.token, timeout=self.timeout_seconds)
        return self._pro
