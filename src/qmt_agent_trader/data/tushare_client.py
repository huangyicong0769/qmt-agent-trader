"""Tushare Pro client wrapper.

The wrapper keeps token handling out of call sites and makes parameter
construction testable without contacting Tushare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qmt_agent_trader.core.errors import ConfigurationError

DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
    "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
    "free_share,total_mv,circ_mv"
)
INCOME_FIELDS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,"
    "diluted_eps,total_revenue,revenue,operate_profit,total_profit,n_income,"
    "n_income_attr_p,update_flag"
)
BALANCESHEET_FIELDS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_assets,"
    "total_liab,total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int,update_flag"
)
CASHFLOW_FIELDS = (
    "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,net_profit,"
    "c_fr_sale_sg,n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,update_flag"
)
FINA_INDICATOR_FIELDS = (
    "ts_code,ann_date,end_date,eps,dt_eps,total_revenue_ps,revenue_ps,"
    "capital_rese_ps,surplus_rese_ps,undist_profit_ps,extra_item,profit_dedt,"
    "gross_margin,current_ratio,quick_ratio,cash_ratio,ar_turn,ca_turn,fa_turn,"
    "assets_turn,roe,roe_dt,roa,roic,debt_to_assets,assets_to_eqt,update_flag"
)
DIVIDEND_FIELDS = (
    "ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,"
    "cash_div,cash_div_tax,record_date,ex_date,pay_date"
)


def _normalize_tushare_date(value: str) -> str:
    return str(value).replace("-", "")


@dataclass(frozen=True)
class TushareRequest:
    api_name: str
    params: dict[str, Any]
    fields: str | None = None


class TushareClient:
    def __init__(self, token: str | None, *, timeout_seconds: float = 300.0) -> None:
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._pro: Any | None = None

    def build_daily_request(
        self, *, start_date: str, end_date: str, ts_code: str | None = None
    ) -> TushareRequest:
        params: dict[str, Any] = {
            "start_date": _normalize_tushare_date(start_date),
            "end_date": _normalize_tushare_date(end_date),
        }
        if ts_code:
            params["ts_code"] = ts_code
        return TushareRequest(api_name="daily", params=params)

    def build_daily_by_trade_date_request(self, trade_date: str) -> TushareRequest:
        return TushareRequest(
            api_name="daily", params={"trade_date": _normalize_tushare_date(trade_date)}
        )

    def build_fund_daily_request(
        self, *, ts_code: str, start_date: str, end_date: str
    ) -> TushareRequest:
        return TushareRequest(
            api_name="fund_daily",
            params={
                "ts_code": ts_code,
                "start_date": _normalize_tushare_date(start_date),
                "end_date": _normalize_tushare_date(end_date),
            },
        )

    def build_trade_calendar_request(self, *, start_date: str, end_date: str) -> TushareRequest:
        return TushareRequest(
            api_name="trade_cal",
            params={
                "exchange": "SSE",
                "start_date": _normalize_tushare_date(start_date),
                "end_date": _normalize_tushare_date(end_date),
            },
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
            params={
                "start_date": _normalize_tushare_date(start_date),
                "end_date": _normalize_tushare_date(end_date),
            },
            fields="ts_code,trade_date,suspend_type",
        )

    def build_stk_limit_by_trade_date_request(self, trade_date: str) -> TushareRequest:
        return TushareRequest(
            api_name="stk_limit",
            params={"trade_date": _normalize_tushare_date(trade_date)},
            fields="ts_code,trade_date,up_limit,down_limit",
        )

    def build_daily_basic_request(
        self,
        *,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        ts_code: str | None = None,
    ) -> TushareRequest:
        params = _date_params(
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            ts_code=ts_code,
        )
        return TushareRequest(api_name="daily_basic", params=params, fields=DAILY_BASIC_FIELDS)

    def build_income_request(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        ts_code: str | None = None,
    ) -> TushareRequest:
        return self._build_financial_request(
            "income",
            fields=INCOME_FIELDS,
            start_date=start_date,
            end_date=end_date,
            period=period,
            ts_code=ts_code,
        )

    def build_balancesheet_request(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        ts_code: str | None = None,
    ) -> TushareRequest:
        return self._build_financial_request(
            "balancesheet",
            fields=BALANCESHEET_FIELDS,
            start_date=start_date,
            end_date=end_date,
            period=period,
            ts_code=ts_code,
        )

    def build_cashflow_request(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        ts_code: str | None = None,
    ) -> TushareRequest:
        return self._build_financial_request(
            "cashflow",
            fields=CASHFLOW_FIELDS,
            start_date=start_date,
            end_date=end_date,
            period=period,
            ts_code=ts_code,
        )

    def build_fina_indicator_request(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        ts_code: str | None = None,
    ) -> TushareRequest:
        return self._build_financial_request(
            "fina_indicator",
            fields=FINA_INDICATOR_FIELDS,
            start_date=start_date,
            end_date=end_date,
            period=period,
            ts_code=ts_code,
        )

    def build_dividend_request(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        ts_code: str | None = None,
    ) -> TushareRequest:
        return self._build_financial_request(
            "dividend",
            fields=DIVIDEND_FIELDS,
            start_date=start_date,
            end_date=end_date,
            period=period,
            ts_code=ts_code,
        )

    def build_macro_request(
        self,
        *,
        api_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        fields: str | None = None,
        **params: Any,
    ) -> TushareRequest:
        request_params = {
            key: value
            for key, value in params.items()
            if value is not None
        }
        if start_date is not None:
            request_params["start_date"] = _normalize_tushare_date(start_date)
        if end_date is not None:
            request_params["end_date"] = _normalize_tushare_date(end_date)
        return TushareRequest(api_name=api_name, params=request_params, fields=fields)

    def _build_financial_request(
        self,
        api_name: str,
        *,
        fields: str,
        start_date: str | None,
        end_date: str | None,
        period: str | None,
        ts_code: str | None,
    ) -> TushareRequest:
        params = _date_params(
            start_date=start_date,
            end_date=end_date,
            period=period,
            ts_code=ts_code,
        )
        return TushareRequest(api_name=api_name, params=params, fields=fields)

    def pro(self) -> Any:
        if not self.token:
            raise ConfigurationError("TUSHARE_TOKEN is required for live Tushare requests")
        if self._pro is None:
            import tushare as ts

            ts.set_token(self.token)
            self._pro = ts.pro_api(self.token, timeout=self.timeout_seconds)
        return self._pro

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        api = self.pro()
        kwargs = dict(request.params)
        if request.fields is not None:
            kwargs["fields"] = request.fields
        return api.query(request.api_name, **kwargs)


def _date_params(**values: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if key.endswith("date") or key == "period":
            params[key] = _normalize_tushare_date(value)
        else:
            params[key] = value
    return params
