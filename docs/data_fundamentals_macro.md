# Fundamentals and Macro Data

This project keeps remote Tushare data in raw Parquet datasets first, then
builds point-in-time query views from those raw files.

## Fundamentals

Supported raw datasets:

- `tushare_daily_basic`
- `tushare_income`
- `tushare_balancesheet`
- `tushare_cashflow`
- `tushare_fina_indicator`
- `tushare_dividend`

Update commands:

```bash
uv run qmt-agent data update-fundamentals --from 20200101 --to 20260630 --dry-run
uv run qmt-agent data update-fundamentals --from 20200101 --to 20260630 --ts-code 000001.SZ
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

These factors require `tushare_daily_basic` and PIT financial datasets. They are
computed from a factor context, not by adding fundamentals to canonical bars.

## Macro

Supported macro dataset registry:

- `cn_gdp`
- `cn_cpi`
- `cn_ppi`
- `shibor`

Update commands:

```bash
uv run qmt-agent data update-macro --from 20200101 --to 20260630 --datasets cn_cpi,cn_ppi,shibor --dry-run
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
