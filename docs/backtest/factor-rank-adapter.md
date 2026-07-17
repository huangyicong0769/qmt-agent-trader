# Factor-rank research adapter

The canonical adapter is a deterministic, research-only backend for
`FACTOR_RANK_LONG_ONLY`. It never authorizes live trading.

## Timeline and valuation

Signals are computed after the signal-date close. Execution occurs after at least one
trading session, at the execution-date open. Opening equity and target quantities use
only current execution opens; end-of-day equity uses only current closes. A missing,
non-finite, or non-positive required current price aborts the run with
`BacktestDataIntegrityError`. There is no stale-close, zero-value, or synthetic-price
fallback.

Weekly and monthly signal dates are the final available trading session in their ISO
week or calendar month. The declared delay is counted in trading sessions.

Rolling-universe snapshots include the first in-range trading session as an anchor and
then the final available session of each configured week or month. Each signal resolves
the latest membership snapshot on or before its signal date. A missing initial snapshot
or an empty as-of membership raises a typed universe-integrity error; neither condition
is converted into an empty selection.

The adapter resolves the maximum declared lookback across all requested factors and
loads that many prior open sessions before the requested start. Insufficient calendar
history raises `INSUFFICIENT_FACTOR_WARMUP_HISTORY`. Warm-up bars feed factor
calculation only: execution schedules, trades, equity points, and performance metrics
remain confined to the requested session window. Report metadata separately identifies
the loaded panel start and the performance start/end. Calendar evidence must also be
continuous from the first warm-up session through the requested end. A missing whole
panel session raises `MISSING_FACTOR_WARMUP_SESSION`; per-symbol shortfalls are exposed
as `insufficient_history_by_symbol` and the symbol is excluded from ranking until its
required history exists. Factor coverage, IC, and walk-forward diagnostics use only
performance dates, and reports disclose the diagnostic window and excluded warm-up row
counts.

## Portfolio semantics

When a `StrategySpec` exists, it is the sole authority for factor identity and
direction, portfolio construction, rebalance behavior, execution delay, and
slippage. `StrategyBacktestConfig` only transports those values to the runtime.
Any conflicting transport value blocks with `CONFIG_SPEC_MISMATCH` before universe
resolution, cache lookup, factor computation, or market-data loading.
`rebalance_frequency` is therefore a strategy-semantic value and must match the spec.
`universe_rebalance_frequency` is the independent rolling-universe cadence; when it is
omitted, the authoritative strategy frequency is used.

When a strategy ID exists in the Registry, its canonical spec fingerprint and Registry
copy are authoritative. An inline spec cannot replace a saved spec with the same ID;
such a conflict returns `SAVED_STRATEGY_SPEC_MISMATCH`. The config ID must also equal
the inline `StrategySpec.strategy_id`. Effective identity is resolved from the top-level
ID, otherwise the inline spec ID, otherwise a temporary factor strategy ID. Registry
identity, generated-code capability, and temporary factor-spec construction all finish
before universe resolution or cache access.

## Strategy identity modes

Every backtest config declares one identity mode:

- `registry`: reload and verify the saved Registry strategy;
- `inline`: execute the supplied unsaved StrategySpec without Registry lookup;
- `adhoc`: execute a temporary factor baseline without Registry lookup.

Generated ad-hoc IDs are cache/report identifiers only. They can never cause a
saved strategy with the same text ID to be loaded.

Cache schema `factor-rank-v4` uses content SHA-256 values. Governed DataLake
writes persist a sidecar content manifest bound to file size, mtime, ctime, and
inode. A missing or stale manifest triggers one full rehash and refresh;
subsequent cache-key construction reads the small manifest instead of the full
Parquet payload. Strategy specs, saved state, strategy and factor code, resolved
universe payloads, and all selected raw datasets remain part of provenance. The
same manifest is stored in the completed response and report config.

The adapter ranks normalized factor values descending. `lower_is_better` negates the
declared single factor before ranking and IC diagnostics. Existing holdings inside
`top_n + rank_buffer` are retained before new entries fill vacancies. The cash buffer is
excluded from target investment. If planned one-way turnover is below
`min_turnover_threshold`, the entire rebalance is skipped.

Universe resolution preserves ranking order through stable deduplication and
`max_symbols` truncation. Explicit-symbol universes preserve the user-declared order;
only unranked, non-explicit universes are deterministically sorted by symbol.
Ranked candidates use an explicit stable sort, with ascending symbol order breaking
equal primary ranking values.

One-way turnover is half gross traded notional divided by pre-trade equity. Every
rebalance also preserves gross notional and selection Jaccard overlap. Explicit fees and
slippage are separately accumulated. `same_trade_gross_return` adds those realized costs
back to net terminal equity; it is not an independently re-sized zero-cost counterfactual.

Buy affordability is lot-aware and includes the full configured cost breakdown,
including minimum commission. Cash and positions are checked after every trade and at
day end. Negative or non-finite cash, non-positive completed positions, and non-finite or
negative equity raise `BacktestAccountingError`; they never produce partial metrics.

## Data and capability boundaries

Expected open sessions come from `raw/tushare/trade_cal`, independently of observed bar
rows. A completely absent open session raises `MISSING_EXPECTED_TRADING_SESSION`.
Calendar evidence must cover every natural date in the requested interval; absent dates
raise `TRADING_CALENDAR_PARTIAL_COVERAGE` and are never inferred to be closed. Invalid
date/state values and conflicting exchange states raise `TRADING_CALENDAR_INVALID` and
`TRADING_CALENDAR_CONFLICTING_STATE`, respectively.
Missing required symbol-day bars and invalid open/close values raise typed data-integrity
errors; there is no previous-close, zero-price, or synthetic-bar fallback.

Both market bars and computed factors require one row per symbol and trade date.
Identical and conflicting duplicates are rejected as `DUPLICATE_SYMBOL_DATE_BAR` or
`DUPLICATE_FACTOR_SYMBOL_DATE`; symbol lookup never selects an arbitrary first row.
Raw daily-bar duplicates are rejected before normalization, and exact factor-source
duplicates are rejected before joins as `DUPLICATE_EXACT_FACTOR_INPUT`; neither path
uses last-row-wins deduplication.

ASOF factor sources also require unique visible identities. Duplicate
`(symbol, visible_date)` or marketwide `visible_date` values raise
`DUPLICATE_ASOF_VISIBLE_KEY`; storage order is never a tie-break. Universe market-cap
inputs validate daily-basic symbol-date uniqueness before latest-row selection and raise
`DUPLICATE_UNIVERSE_SOURCE_KEY` on conflict.

Trade-state columns are usable only with source evidence. Stock rows require
`raw/tushare/suspend_d`, `raw/tushare/stk_limit`, and historical
`raw/tushare/namechange`; missing required evidence blocks execution with
`TRADE_STATE_SOURCE_NOT_READY`. ETF-only rows require `raw/tushare/stk_limit`.
`stk_limit` must exactly cover every executable stock symbol-date or the run raises
`TRADE_STATE_PARTIAL_COVERAGE`. Sparse suspension rows and non-overlapping historical
name intervals become `False` only after their datasets are proven present, and the
completed panel records source and completeness metadata for every state field.
Stock limit prices must be finite, positive, satisfy `down_limit < up_limit`, and cover
every stock bar. Execution eligibility is opening-only:
`limit_up_at_open` and `limit_down_at_open` compare the execution open with the validated
limits; a close at a limit cannot block an earlier opening trade. Suspension and ST
state come from dated suspension and historical name-change evidence, never the current
company name.

Every normalized bar is tagged `asset_type=stock` or `asset_type=etf`. Stock and ETF
partitions are enriched separately. Both use the endpoint contract's `stk_limit`
opening prices, while ETF rows never use stock-only suspension or name-change/ST
semantics.

The research runner accepts only prevalidated canonical rows containing OHLC, volume,
amount, turnover, and all four opening execution-state fields. Missing state columns
raise `MISSING_EXECUTION_STATE_COLUMNS`, null state raises `UNKNOWN_EXECUTION_STATE`,
and other missing canonical fields raise `MISSING_CANONICAL_BAR_COLUMNS`. The runner
coerces fields that exist but never synthesizes absent numeric values or boolean state.

Abrupt daily cross-sectional coverage collapse blocks broad-universe runs. Universe
limits are optional and any truncation records candidate, pre-limit, selected, effective
limit, and source evidence. The canonical adapter supports only
`execution.cost_model == "a_share_default"` and empty `risk_constraints`. Unsupported
portfolio, timing, transform, cost, risk, or custom semantics return `BLOCKED`.

Generated strategy Python is not executed. Both an explicit request `code_path` and a
saved strategy's registry `code_path` return
`GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED` until a process-isolated runner exists. A
spec-only draft without a code path remains eligible for canonical execution.

## Canonical factor fields

`turnover` is sourced from `tushare/daily_basic.turnover_rate`. A null placeholder
in normalized daily bars is not evidence that the field is available and does not
prevent source resolution.

## Point-in-time universes

A snapshot uses the latest open market session on or before the requested as-of date
and requires a bar on that exact session. Previous-session bars are never carried
forward as current eligibility evidence. Listing eligibility uses `list_date` and
`delist_date`; current `list_status` and current company names are not historical
filters. Historical industry/theme selection blocks without dated classification
records.

Index-weight universes use the latest snapshot on or before the as-of date. Interval
membership uses `in_date <= as_of < out_date`.

## Universe session authority

`trade_cal` is authoritative for snapshot effective sessions and rolling
rebalance dates. The requested as-of date must itself have calendar evidence,
including closed weekend and holiday records. An official open session with no
market-wide bars for the requested asset type raises
`UNIVERSE_MARKET_SESSION_NOT_READY`; it is not reported as a valid empty
universe.

## Liquidity-window completeness

`avg_amount_20d` and `avg_volume_20d` require exactly 20 official sessions and
20 non-null observations for the corresponding field. Short listings, missing
sessions, suspension gaps, and null values leave the metric unavailable and
produce explicit observation-count evidence.

## Point-in-time date and index evidence

Empty `delist_date` and `out_date` values represent open intervals. Non-empty
malformed dates are invalid source evidence and fail closed.

Each requested index code is resolved independently. The resolver uses the
latest `index_weight` snapshot for that code when available, otherwise active
`index_member` intervals. Missing evidence for any requested code raises
`INDEX_MEMBERSHIP_NOT_READY`.

## ETF opening state

ETF opening limit state uses `tushare/stk_limit`. ETF ST state is not applicable, and
presence of a valid exact-session `fund_daily` row is the evidence that the row is
tradable rather than suspended.

## Financial revisions

Financial ASOF fields are reduced by symbol, visible date, report period, update flag,
and actual announcement date before the generic ASOF join. Identical business ranks
with conflicting values fail closed.

New governed reports use schema `2.0` with canonical metrics, diagnostics, dated equity,
rebalance points, trade blotter, data quality, and cost attribution. Legacy `payload`,
`equity_curve`, and `turnover_series` remain for one migration cycle.

Known market-data, universe-timeline, and accounting failures are converted to structured
`ERROR` payloads only at the outer Agent-tool boundary. They do not write a completed
report or enter the successful-result cache. Unexpected software exceptions propagate to
the normal runtime error handler.

Each execution schedule entry is classified before simulation. Missing factor dates,
all-null cross sections, and signals emptied by point-in-time universe filtering create
skipped rebalance records with `factor_signal_date_missing`,
`factor_signal_all_null`, or `factor_signal_empty_after_universe_filter`. A run with no
executable scheduled signal raises `NO_EXECUTABLE_FACTOR_SIGNALS`; a signal window with
no delayed execution session raises `NO_EXECUTION_SESSION_AFTER_SIGNAL`. Completed
results expose scheduled, available, and unavailable signal counts, and still contain
exactly one equity point per expected trading date.
These signal-availability counts are canonical members of `data_quality`; legacy
top-level keys remain temporarily and are sourced from the same canonical values.

Numeric inputs fail validation before simulation. Initial cash must be finite and
positive; top-N must be positive; the position cap must be in `(0, 1]`; the cash buffer
must be in `[0, 1)`; turnover threshold must be in `[0, 1]`; rank buffer and slippage
must be non-negative; execution delay must be at least one trading session; sensitivity
cost multipliers must be positive; and expected trading dates must be non-empty, sorted,
and unique.

## Diagnostics

```powershell
uv run python scripts/diagnose_factor_rank_report.py reports/research/research_<id>.json
```

Direct JSON without a governed manifest is rejected unless the operator explicitly adds
`--unsafe-direct-json` for offline debugging.

Diagnostics consume the same canonical metric map returned in the schema `2.0` result.
If cost drag or average top-N overlap evidence is absent, the corresponding diagnostic is
`NOT_COMPUTED`, never `PASS`.
No comparable pair of completed selections produces `average_top_n_overlap: null`; it is
never converted to a fabricated `0.0`.
