"""Tushare Pro client wrapper.

The wrapper keeps token handling out of call sites and makes parameter
construction testable without contacting Tushare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qmt_agent_trader.core.errors import ConfigurationError


@dataclass(frozen=True)
class TushareRequest:
    api_name: str
    params: dict[str, Any]
    fields: str | None = None


class TushareClient:
    def __init__(self, token: str | None) -> None:
        self.token = token
        self._pro: Any | None = None

    def build_daily_request(
        self, *, start_date: str, end_date: str, ts_code: str | None = None
    ) -> TushareRequest:
        params: dict[str, Any] = {"start_date": start_date, "end_date": end_date}
        if ts_code:
            params["ts_code"] = ts_code
        return TushareRequest(api_name="daily", params=params)

    def build_daily_by_trade_date_request(self, trade_date: str) -> TushareRequest:
        return TushareRequest(api_name="daily", params={"trade_date": trade_date})

    def build_trade_calendar_request(self, *, start_date: str, end_date: str) -> TushareRequest:
        return TushareRequest(
            api_name="trade_cal",
            params={"exchange": "SSE", "start_date": start_date, "end_date": end_date},
        )

    def build_stock_basic_request(self) -> TushareRequest:
        return TushareRequest(
            api_name="stock_basic",
            params={"exchange": "", "list_status": "L"},
            fields="ts_code,symbol,name,area,industry,list_date",
        )

    def build_namechange_request(
        self, *, limit: int | None = None, offset: int | None = None
    ) -> TushareRequest:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return TushareRequest(
            api_name="namechange",
            params=params,
            fields="ts_code,name,start_date,end_date,change_reason",
        )

    def build_etf_basic_request(self) -> TushareRequest:
        return TushareRequest(
            api_name="fund_basic",
            params={"market": "E", "status": "L"},
            fields="ts_code,name,management,custodian,fund_type,found_date,due_date,list_date",
        )

    def build_suspend_request(self, *, start_date: str, end_date: str) -> TushareRequest:
        return TushareRequest(
            api_name="suspend_d",
            params={"start_date": start_date, "end_date": end_date},
            fields="ts_code,trade_date,suspend_type",
        )

    def build_stk_limit_by_trade_date_request(self, trade_date: str) -> TushareRequest:
        return TushareRequest(
            api_name="stk_limit",
            params={"trade_date": trade_date},
            fields="ts_code,trade_date,up_limit,down_limit",
        )

    def pro(self) -> Any:
        if not self.token:
            raise ConfigurationError("TUSHARE_TOKEN is required for live Tushare requests")
        if self._pro is None:
            import tushare as ts

            ts.set_token(self.token)
            self._pro = ts.pro_api(self.token)
        return self._pro

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        api = self.pro()
        kwargs = dict(request.params)
        if request.fields is not None:
            kwargs["fields"] = request.fields
        return api.query(request.api_name, **kwargs)
