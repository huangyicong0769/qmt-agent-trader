# Factor-rank backtest correctness repair

This repository implementation record corresponds to the reviewed 13-task plan dated
2026-07-13. The repair replaces rebalance-only valuation with a strict daily ledger,
propagates declared strategy semantics, blocks unsupported/generated-code fallbacks,
adds data-quality and universe evidence, attributes churn and costs, publishes schema-v2
reports, and renders real dated equity in the UI.

The authoritative implemented semantics and operator workflow are documented in
[`docs/backtest/factor-rank-adapter.md`](../../backtest/factor-rank-adapter.md). Atomic
commits on the implementation branch preserve the task-by-task delivery history and its
focused regressions.

Two contradictions in the source plan were resolved in favor of its global fail-closed
constraints: current-price gaps never fall back to previous closes, and a held-symbol bar
gap produces a typed failed run rather than a completed report with partial metrics.
