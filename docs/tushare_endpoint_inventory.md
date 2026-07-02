# Tushare Endpoint Inventory

This inventory was extracted from the official Tushare documentation visible in the user's
browser session and the official `tushare.pro/wctapi/documents/*.md` pages on 2026-07-02.
It intentionally records endpoint names, parameters, fields, keys, storage intent, and PIT
rules only. It does not include or require any Tushare token, cookie, or session value.

If a candidate endpoint page was not available during extraction it is marked
`DOC_UNAVAILABLE` in `src/qmt_agent_trader/data/providers/tushare/endpoints.yml` and is
registered as a placeholder.

## Implemented endpoints

| api_name | doc | category | key columns | ts_code | date parameters | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `daily` | https://tushare.pro/document/2?doc_id=27 | market | `ts_code, trade_date` | yes | `trade_date, start_date, end_date` | Stock daily OHLCV. |
| `daily_basic` | https://tushare.pro/document/2?doc_id=32 | market | `ts_code, trade_date` | yes | `trade_date, start_date, end_date` | Daily valuation and share metrics. |
| `suspend_d` | https://tushare.pro/document/2?doc_id=31 | market | `ts_code, trade_date` | yes | `suspend_date, resume_date` | Suspension status. |
| `stk_limit` | https://tushare.pro/document/2?doc_id=183 | market | `ts_code, trade_date` | yes | `trade_date, start_date, end_date` | Daily limit-up/down prices. |
| `fund_basic` | https://tushare.pro/document/2?doc_id=19 | fund | `ts_code` | yes | none | Fund and ETF master data. |
| `fund_daily` | https://tushare.pro/document/2?doc_id=127 | fund | `ts_code, trade_date` | yes | `trade_date, start_date, end_date` | Fund/ETF daily OHLCV. |
| `index_basic` | https://tushare.pro/document/2?doc_id=94 | index | `ts_code` | yes | none | Index master data. |
| `index_daily` | https://tushare.pro/document/2?doc_id=95 | index | `ts_code, trade_date` | yes | `trade_date, start_date, end_date` | Index daily OHLCV. |
| `stock_basic` | https://tushare.pro/document/2?doc_id=25 | security | `ts_code` | yes | none | Stock master data. |
| `namechange` | https://tushare.pro/document/2?doc_id=100 | security | `ts_code, start_date, name` | yes | `start_date, end_date` | Security name-change history; also modeled as corporate action events. |
| `trade_cal` | https://tushare.pro/document/2?doc_id=26 | security | `exchange, cal_date` | no | `start_date, end_date` | Trading calendar. |
| `income` | https://tushare.pro/document/2?doc_id=33 | fundamental | `ts_code, end_date, ann_date, report_type` | yes | `start_date, end_date, period` | PIT visible date is `f_ann_date` if present else `ann_date`. |
| `balancesheet` | https://tushare.pro/document/2?doc_id=36 | fundamental | `ts_code, end_date, ann_date, report_type` | yes | `start_date, end_date, period` | PIT visible date is `f_ann_date` if present else `ann_date`. |
| `cashflow` | https://tushare.pro/document/2?doc_id=44 | fundamental | `ts_code, end_date, ann_date, report_type` | yes | `start_date, end_date, period` | PIT visible date is `f_ann_date` if present else `ann_date`. |
| `fina_indicator` | https://tushare.pro/document/2?doc_id=79 | fundamental | `ts_code, end_date, ann_date` | yes | `start_date, end_date, period` | Financial indicators, PIT by `ann_date`. |
| `dividend` | https://tushare.pro/document/2?doc_id=103 | corporate_action | `ts_code, end_date, ann_date, div_proc` | yes | `ann_date, record_date, ex_date, imp_ann_date` | Stored as corporate action events, not financial reports. |
| `cn_gdp` | https://tushare.pro/document/2?doc_id=227 | macro | `quarter` | no | `q, start_q, end_q` | Quarterly macro series. |
| `cn_cpi` | https://tushare.pro/document/2?doc_id=228 | macro | `month` | no | `m, start_m, end_m` | Monthly macro series. |
| `cn_ppi` | https://tushare.pro/document/2?doc_id=245 | macro | `month` | no | `m, start_m, end_m` | Monthly macro series. |
| `shibor` | https://tushare.pro/document/2?doc_id=149 | macro | `date` | no | `date, start_date, end_date` | Daily money-market rate series. |

## Placeholder endpoints

The following endpoints are registered for planner visibility but are not executable in this
refactor: `adj_factor`, `moneyflow`, `margin`, `top10_holders`, `concept`, `index_weight`,
`index_dailybasic`, `index_member`, `fund_nav`, `fund_portfolio`, `repurchase`,
`share_float`, and `stk_holdernumber`.
