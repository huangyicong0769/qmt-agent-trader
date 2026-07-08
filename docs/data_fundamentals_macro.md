# Fundamentals and Macro Data

This project keeps remote Tushare data in raw Parquet datasets first, then
builds point-in-time query views from those raw files.

## Fundamentals

Supported raw datasets:

- `tushare/daily_basic`
- `tushare/income`
- `tushare/balancesheet`
- `tushare/cashflow`
- `tushare/fina_indicator`
- `tushare/dividend`

Fetch and build commands:

```bash
uv run qmt-agent data capabilities --category fundamental
uv run qmt-agent data plan-fetch --api daily_basic --symbols 000001.SZ --from 20200101 --to 20260630 --fields ts_code,trade_date,pe_ttm,pb,total_mv
uv run qmt-agent data fetch --api daily_basic --symbols 000001.SZ --from 20200101 --to 20260630 --fields ts_code,trade_date,pe_ttm,pb,total_mv --execute-plan
uv run qmt-agent data build-table --table financial_reports_wide
uv run qmt-agent data build-table --table financial_current_wide --snapshot-as-of-date 20260630
uv run qmt-agent data validate-fundamentals
```

Agent query:

```bash
uv run qmt-agent agent call-tool \
  --name query_fundamentals_pit \
  --params '{"symbols":["000001.SZ"],"as_of_date":"20240131","fields":["pe_ttm","pb","roe"]}'
```

PIT rules:

- `daily_basic` snapshots use `trade_date <= as_of_date`.
- Financial statements use `visible_date <= as_of_date`.
- `visible_date` is `f_ann_date` when present, otherwise `ann_date`.
- Rows without either announcement date are marked not PIT-safe and are not used
  by strict fundamentals queries.
- The query output includes metadata with `point_in_time`, `pit_rule`,
  `datasets_used`, `missing_symbols`, and `missing_fields`.

Fundamental factors:

- Value: `size_log_mktcap`, `pe_ttm_rank`, `pb_rank`, `dividend_yield`
- Quality: `roe_rank`, `gross_margin_rank`, `debt_to_assets_rank`

These factors require `tushare/daily_basic` and PIT financial datasets. They are
computed from a factor context, not by adding fundamentals to canonical bars.

## Macro

Supported macro dataset registry:

- `cn_gdp`
- `cn_cpi`
- `cn_ppi`
- `shibor`

Fetch and build commands:

```bash
uv run qmt-agent data capabilities --category macro
uv run qmt-agent data plan-fetch --api cn_cpi --from 20200101 --to 20260630
uv run qmt-agent data fetch --api cn_cpi --from 20200101 --to 20260630 --execute-plan
uv run qmt-agent data build-table --table macro_series
uv run qmt-agent data validate-macro
```

Agent query:

```bash
uv run qmt-agent agent call-tool \
  --name query_macro_series_pit \
  --params '{"dataset":"shibor","as_of_date":"20240131","start_date":"20240101","fields":["on"],"strict_pit":true}'
```

Macro PIT limitations:

- `shibor` is treated as PIT-safe with same-day visibility.
- Monthly macro datasets use a conservative `period_end + 15 calendar days`
  visibility approximation.
- Quarterly macro datasets use a conservative `period_end + 45 calendar days`
  visibility approximation.
- Datasets using conservative visibility are returned as `PIT_NOT_VALIDATED`
  when `strict_pit=true`.
- MCP or Tavily macro/news search is explanatory research only and must not be
  mixed into backtest data.

Common failures:

- Tushare permission is missing for a requested API.
- The local data lake has no raw dataset for the query.
- Requested fields are not present in the returned Tushare schema.
- Financial rows have no announcement date and are excluded from strict PIT use.
