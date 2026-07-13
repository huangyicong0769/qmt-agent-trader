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

## Data and capability boundaries

Abrupt daily cross-sectional coverage collapse blocks broad-universe runs. Universe
limits are optional and any truncation records candidate, pre-limit, selected, effective
limit, and source evidence. Unsupported portfolio, timing, transform, or custom semantics
return `BLOCKED`. Generated strategy Python is not executed; a code path returns
`GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED` until a process-isolated runner exists.

New governed reports use schema `2.0` with canonical metrics, diagnostics, dated equity,
rebalance points, trade blotter, data quality, and cost attribution. Legacy `payload`,
`equity_curve`, and `turnover_series` remain for one migration cycle.

## Diagnostics

```powershell
uv run python scripts/diagnose_factor_rank_report.py reports/research/research_<id>.json
```

Direct JSON without a governed manifest is rejected unless the operator explicitly adds
`--unsafe-direct-json` for offline debugging.
