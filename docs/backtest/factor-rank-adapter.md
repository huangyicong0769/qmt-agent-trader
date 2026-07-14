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

## Portfolio semantics

The adapter ranks normalized factor values descending. `lower_is_better` negates the
declared single factor before ranking and IC diagnostics. Existing holdings inside
`top_n + rank_buffer` are retained before new entries fill vacancies. The cash buffer is
excluded from target investment. If planned one-way turnover is below
`min_turnover_threshold`, the entire rebalance is skipped.

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
Missing required symbol-day bars and invalid open/close values raise typed data-integrity
errors; there is no previous-close, zero-price, or synthetic-bar fallback.

Abrupt daily cross-sectional coverage collapse blocks broad-universe runs. Universe
limits are optional and any truncation records candidate, pre-limit, selected, effective
limit, and source evidence. The canonical adapter supports only
`execution.cost_model == "a_share_default"` and empty `risk_constraints`. Unsupported
portfolio, timing, transform, cost, risk, or custom semantics return `BLOCKED`.

Generated strategy Python is not executed. Both an explicit request `code_path` and a
saved strategy's registry `code_path` return
`GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED` until a process-isolated runner exists. A
spec-only draft without a code path remains eligible for canonical execution.

New governed reports use schema `2.0` with canonical metrics, diagnostics, dated equity,
rebalance points, trade blotter, data quality, and cost attribution. Legacy `payload`,
`equity_curve`, and `turnover_series` remain for one migration cycle.

Known market-data, universe-timeline, and accounting failures are converted to structured
`ERROR` payloads only at the outer Agent-tool boundary. They do not write a completed
report or enter the successful-result cache. Unexpected software exceptions propagate to
the normal runtime error handler.

## Diagnostics

```powershell
uv run python scripts/diagnose_factor_rank_report.py reports/research/research_<id>.json
```

Direct JSON without a governed manifest is rejected unless the operator explicitly adds
`--unsafe-direct-json` for offline debugging.

Diagnostics consume the same canonical metric map returned in the schema `2.0` result.
If cost drag or average top-N overlap evidence is absent, the corresponding diagnostic is
`NOT_COMPUTED`, never `PASS`.
