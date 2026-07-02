"""Thin generic Tushare query wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd

from qmt_agent_trader.data.tushare_client import TushareClient as LegacyTushareClient


class TushareClient(LegacyTushareClient):
    """Generic client for registry/planner driven Tushare calls.

    The legacy base class still carries short-term request-builder shims for old callers.
    New provider code only uses `query`.
    """

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
