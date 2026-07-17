# Factor-Rank Factor and Universe Correctness Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining factor-value, field-mapping, historical-universe, ETF, strategy-identity, financial-ASOF, and cache-provenance correctness blockers on `codex/factor-rank-backtest-correctness`.

**Architecture:** Keep the factor-rank runner strict and move source-specific semantics into focused normalization layers. Canonical factor fields are resolved from explicit source aliases, historical universes are built from exact point-in-time sessions and dated metadata, and index/financial snapshots use documented business keys rather than storage order. Stock and ETF execution-state enrichment share the validated opening-limit source but retain asset-specific ST and suspension semantics.

**Tech Stack:** Python 3.11+, pandas, NumPy, Pydantic v2, pytest, existing `DataLake`, `FactorRegistry`, `UniverseResolver`, Tushare endpoint registry, Ruff, mypy, and `uv`.

## Global Constraints

- Target branch: `codex/factor-rank-backtest-correctness`; start from its current head.
- Save this plan in the repository as `docs/superpowers/plans/2026-07-14-factor-rank-backtest-factor-universe-repairs.md`.
- Use TDD for every task: failing regression, focused implementation, passing regression, focused commit.
- Use one focused commit per task.
- Preserve `research_only=True` and `live_trading_allowed=False`.
- Do not use current company names or current `list_status` as historical eligibility evidence.
- Do not reuse a stale previous-session bar as evidence that a symbol is eligible on an as-of session.
- Do not resolve business-key conflicts by input order, `keep="first"`, `keep="last"`, or an incomplete sort key.
- Pure `factor_name` requests are ad-hoc research requests and must not read a saved strategy unless the caller explicitly supplies a strategy identity.
- Canonical `turnover` means Tushare `daily_basic.turnover_rate`; do not synthesize it from volume or create a zero-filled substitute.
- ETF support remains in scope. Use `stk_limit` for ETF opening limit prices because the repository endpoint contract declares fund coverage; do not apply stock-only ST/name-change semantics to ETFs.
- Integrity failures create no completed report and no successful cache entry.
- Only typed, expected domain errors are converted at the Agent boundary. Unexpected programming errors propagate.
- Do not add or modify GitHub Actions.
- Do not add a historical extreme-drawdown replay.
- Do not create a new full DataLake-to-Agent end-to-end fixture solely for this repair.
- No new runtime dependency.

---

## File Responsibility Map

### New files

- `src/qmt_agent_trader/factors/source_aliases.py`
  Canonical factor-field aliases and source-field resolution.

- `src/qmt_agent_trader/universe/pit_metadata.py`
  Point-in-time listing metadata, historical classification guards, and index membership normalization.

- `tests/unit/factors/test_price_volume_group_boundaries.py`
  Cross-symbol rolling-window regressions.

- `tests/unit/factors/test_canonical_turnover_source.py`
  `daily_basic.turnover_rate -> turnover` integration at the input-panel boundary.

- `tests/unit/agent/test_adhoc_factor_strategy_identity.py`
  Pure factor requests cannot collide with saved strategy IDs.

- `tests/unit/universe/test_pit_security_master.py`
  Listing/delisting and current-name/list-status leakage regressions.

- `tests/unit/universe/test_exact_session_resolution.py`
  No stale-bar universe membership and bounded lookback loading.

- `tests/unit/universe/test_index_membership_asof.py`
  Latest index-weight snapshot and effective-interval index-member behavior.

- `tests/unit/data/test_etf_opening_trade_state.py`
  ETF opening-limit enrichment without stock ST semantics.

- `tests/unit/factors/test_financial_asof_resolution.py`
  Financial revision tie-break and true ambiguity failure.

- `tests/unit/persistence/test_provenance_content_hash.py`
  Same-size, preserved-mtime file replacement invalidates fingerprints.

### Existing files to modify

- `src/qmt_agent_trader/factors/library/price_volume.py`
- `src/qmt_agent_trader/factors/input_panel.py`
- `src/qmt_agent_trader/data/field_sources.py`
- `src/qmt_agent_trader/data/bars.py`
- `src/qmt_agent_trader/data/trade_state.py`
- `src/qmt_agent_trader/data/trading_calendar.py`
- `src/qmt_agent_trader/data/fundamentals.py`
- `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- `src/qmt_agent_trader/universe/resolver.py`
- `src/qmt_agent_trader/persistence/provenance.py`
- `docs/backtest/factor-rank-adapter.md`

---

### Task 1: Fix Cross-Symbol Rolling Factor Contamination

**Files:**
- Modify: `src/qmt_agent_trader/factors/library/price_volume.py`
- Create: `tests/unit/factors/test_price_volume_group_boundaries.py`

**Interfaces:**
- Produces: `_grouped_rolling_std(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series`
- Produces: `volatility_20d(frame: pd.DataFrame) -> pd.Series` with windows isolated by `symbol`.

- [ ] **Step 1: Write the failing two-symbol regression**

Create `tests/unit/factors/test_price_volume_group_boundaries.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from qmt_agent_trader.factors.library.price_volume import volatility_20d


def test_volatility_window_never_crosses_symbol_boundary() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 25 + ["B"] * 25,
            "trade_date": [*dates, *dates],
            "close": [
                *[100.0 + float(index) for index in range(25)],
                *[10.0 * (2.0**index) for index in range(25)],
            ],
        }
    )

    observed = volatility_20d(frame)
    expected = (
        frame.groupby("symbol", sort=False)["close"]
        .pct_change()
        .groupby(frame["symbol"], sort=False)
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    pd.testing.assert_series_equal(observed, expected, check_names=False)
    assert np.isnan(observed.iloc[25:45]).all()
```

- [ ] **Step 2: Run the test and confirm the current implementation fails**

```bash
uv run pytest tests/unit/factors/test_price_volume_group_boundaries.py -q
```

Expected: FAIL because the current trailing `.rolling(20)` spans the concatenated Series.

- [ ] **Step 3: Implement one grouped rolling helper**

Replace `volatility_20d()` in `price_volume.py` with:

```python
def _grouped_rolling_std(
    values: pd.Series,
    symbols: pd.Series,
    window: int,
) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def volatility_20d(frame: pd.DataFrame) -> pd.Series:
    returns = frame.groupby("symbol", sort=False)["close"].pct_change()
    return _grouped_rolling_std(returns, frame["symbol"], 20)
```

Do not change `ddof`; preserve pandas' existing sample-standard-deviation behavior.

- [ ] **Step 4: Run the focused factor tests**

```bash
uv run pytest \
  tests/unit/factors/test_price_volume_group_boundaries.py \
  tests/unit/test_factor_service.py \
  tests/unit/test_factor_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qmt_agent_trader/factors/library/price_volume.py \
        tests/unit/factors/test_price_volume_group_boundaries.py
git commit -m "fix(factors): isolate volatility windows by symbol"
```

---

### Task 2: Resolve Canonical Turnover from `daily_basic.turnover_rate`

**Files:**
- Create: `src/qmt_agent_trader/factors/source_aliases.py`
- Modify: `src/qmt_agent_trader/data/field_sources.py`
- Modify: `src/qmt_agent_trader/factors/input_panel.py`
- Modify: `src/qmt_agent_trader/data/bars.py`
- Create: `tests/unit/factors/test_canonical_turnover_source.py`
- Modify: `tests/unit/factors/test_input_panel.py`

**Interfaces:**
- Produces: `CanonicalFieldSource`.
- Produces: `resolve_canonical_field_source(index, field, target_frequency) -> CanonicalFieldSource | None`.
- Canonical alias: `turnover -> tushare/daily_basic.turnover_rate`.
- A present but unusable placeholder column does not prevent source resolution.

- [ ] **Step 1: Write the failing canonical-turnover test**

Create `tests/unit/factors/test_canonical_turnover_source.py`:

```python
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.input_panel import build_target_frequency_panel


def _write_sources(lake: DataLake) -> None:
    daily_rows: list[dict[str, object]] = []
    basic_rows: list[dict[str, object]] = []
    limit_rows: list[dict[str, object]] = []
    for offset in range(25):
        day = date(2024, 1, 1) + timedelta(days=offset)
        key = f"{day:%Y%m%d}"
        daily_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": key,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "vol": 1000.0,
                "amount": 10000.0,
            }
        )
        basic_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": key,
                "turnover_rate": float(offset + 1),
            }
        )
        limit_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": key,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        )
    lake.write_parquet(pd.DataFrame(daily_rows), "raw", "tushare/daily")
    lake.write_parquet(pd.DataFrame(basic_rows), "raw", "tushare/daily_basic")
    lake.write_parquet(pd.DataFrame(limit_rows), "raw", "tushare/stk_limit")
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )


def test_canonical_turnover_comes_from_daily_basic_turnover_rate(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    _write_sources(lake)

    panel, metadata = build_target_frequency_panel(
        lake,
        target_frequency=Frequency.DAILY,
        target_start="20240101",
        target_end="20240125",
        required_fields=["turnover"],
        symbols=["000001.SZ"],
    )

    assert metadata["field_sources"]["turnover"]["api_name"] == "daily_basic"
    assert metadata["field_sources"]["turnover"]["source_field"] == "turnover_rate"
    assert panel["turnover"].tolist() == [float(index) for index in range(1, 26)]
```

- [ ] **Step 2: Run the regression**

```bash
uv run pytest tests/unit/factors/test_canonical_turnover_source.py -q
```

Expected: FAIL because the all-null canonical placeholder causes the panel builder to skip source discovery.

- [ ] **Step 3: Add canonical field aliases**

Create `src/qmt_agent_trader/factors/source_aliases.py`:

```python
"""Canonical factor fields backed by differently named raw fields."""

from __future__ import annotations

from dataclasses import dataclass

from qmt_agent_trader.data.field_sources import FieldSourceIndex, FieldSourceSpec
from qmt_agent_trader.data.frequency import Frequency


@dataclass(frozen=True)
class CanonicalFieldSource:
    canonical_field: str
    source_field: str
    source: FieldSourceSpec


_CANONICAL_ALIASES: dict[str, tuple[str, str]] = {
    "turnover": ("daily_basic", "turnover_rate"),
}


def resolve_canonical_field_source(
    index: FieldSourceIndex,
    field: str,
    *,
    target_frequency: Frequency,
) -> CanonicalFieldSource | None:
    alias = _CANONICAL_ALIASES.get(field)
    if alias is None:
        source = index.best_source_for_field(
            field,
            target_frequency=target_frequency,
        )
        return (
            None
            if source is None
            else CanonicalFieldSource(field, field, source)
        )
    api_name, source_field = alias
    source = index.best_source_for_field(
        source_field,
        target_frequency=target_frequency,
        preferred_api=api_name,
    )
    return (
        None
        if source is None
        else CanonicalFieldSource(field, source_field, source)
    )
```

- [ ] **Step 4: Let unusable placeholders fall through to source resolution**

In `input_panel.py`, add:

```python
from qmt_agent_trader.data.bars import column_quality, load_daily_bars
from qmt_agent_trader.factors.source_aliases import resolve_canonical_field_source
```

Add:

```python
def _panel_field_requires_resolution(panel: pd.DataFrame, field: str) -> bool:
    if field not in panel.columns:
        return True
    quality = column_quality(panel, field)
    if quality.get("usable_for_factor") is False:
        return True
    return bool(panel[field].isna().all())
```

Replace:

```python
if field in panel.columns:
    continue
source = source_index.best_source_for_field(...)
```

with:

```python
if not _panel_field_requires_resolution(panel, field):
    continue
resolved = resolve_canonical_field_source(
    source_index,
    field,
    target_frequency=target_frequency,
)
if resolved is None:
    _record_unresolved_source(
        metadata,
        field,
        source_index.sources_for_field(field),
    )
    continue
source = resolved.source
source_field = resolved.source_field
metadata["field_sources"][field] = {
    **source.as_metadata(),
    "source_field": source_field,
    "canonical_field": field,
}
```

- [ ] **Step 5: Parameterize exact and ASOF joins by source field**

Change both join helper signatures:

```python
def _join_exact_field(
    ...,
    field: str,
    source_field: str,
    ...,
) -> pd.DataFrame:
```

```python
def _join_asof_snapshot_field(
    ...,
    field: str,
    source_field: str,
    ...,
) -> pd.DataFrame:
```

In `_join_exact_field()`, read `source_field`, validate it, then rename it before merging:

```python
columns = _source_read_columns(source, source_field)
...
missing_columns = [
    column
    for column in (
        source.entity_column,
        source.visible_time_column,
        source_field,
    )
    if column is not None and column not in raw.columns
]
...
data = raw.rename(
    columns={
        source.entity_column: "symbol",
        source_field: field,
    }
).copy()
```

Apply the same `source_field -> field` rename before ASOF resolution.

Before adding a resolved field, remove the unusable placeholder:

```python
panel = panel.drop(columns=[field], errors="ignore")
```

- [ ] **Step 6: Preserve the placeholder only as quality metadata**

Keep the canonical `turnover` column in `normalize_tushare_daily()` for the runner schema, but retain:

```python
column_quality["turnover"] = {
    "source": "missing_from_raw",
    "imputed": True,
    "usable_for_factor": False,
}
```

Do not fill it with zero and do not mark it usable.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/unit/factors/test_canonical_turnover_source.py \
  tests/unit/factors/test_input_panel.py \
  tests/unit/test_factor_service.py \
  tests/unit/test_data_bars.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/factors/source_aliases.py \
        src/qmt_agent_trader/data/field_sources.py \
        src/qmt_agent_trader/factors/input_panel.py \
        src/qmt_agent_trader/data/bars.py \
        tests/unit/factors/test_canonical_turnover_source.py \
        tests/unit/factors/test_input_panel.py
git commit -m "fix(factors): source canonical turnover from daily basic"
```

---

### Task 3: Separate Ad-Hoc Factor Identity from Saved Strategy Identity

**Files:**
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/agent/test_adhoc_factor_strategy_identity.py`
- Modify: `tests/unit/agent/test_backtest_pre_cache_identity.py`

**Interfaces:**
- Pure `factor_name` request: no Registry lookup.
- Explicit `strategy_id` or inline `strategy_spec.strategy_id`: Registry lookup remains mandatory.
- Produces deterministic ad-hoc ID `adhoc_factor_<factor_id>` for cache stability.

- [ ] **Step 1: Write the collision regression**

Create `tests/unit/agent/test_adhoc_factor_strategy_identity.py`:

```python
from __future__ import annotations

from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySpec, StrategySource
from qmt_agent_trader.core.types import ApprovalStatus


def test_factor_only_request_does_not_load_same_named_saved_strategy(
    wired_strategy_tools,
    monkeypatch,
) -> None:
    saved_spec = StrategySpec.model_validate(
        {
            "strategy_id": "factor_momentum_20d",
            "name": "Saved collision",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "momentum_20d"}],
            "portfolio": {"top_n": 3},
            "rebalance": {"frequency": "monthly"},
        }
    )
    strategy_tools._strategy_registry().save_candidate(
        SavedStrategy(
            strategy_id=saved_spec.strategy_id,
            name=saved_spec.name,
            version=saved_spec.version,
            source=StrategySource.AGENT_GENERATED,
            status=ApprovalStatus.DRAFT,
            spec=saved_spec,
            implementation_ref="spec:draft",
        )
    )

    intent = strategy_tools._resolve_backtest_intent(
        {"factor_name": "momentum_20d"},
        requested_strategy_frequency="weekly",
        requested_top_n=20,
    )

    assert not isinstance(intent, dict)
    assert intent.saved_strategy is None
    assert intent.strategy_id == "adhoc_factor_momentum_20d"
    assert intent.strategy_spec.portfolio.top_n == 20
    assert intent.strategy_spec.rebalance.frequency == "weekly"
```

- [ ] **Step 2: Run the regression**

```bash
uv run pytest tests/unit/agent/test_adhoc_factor_strategy_identity.py -q
```

Expected: FAIL because `factor_momentum_20d` currently participates in Registry lookup.

- [ ] **Step 3: Resolve explicit and ad-hoc identities separately**

In `_resolve_backtest_intent()` replace the effective-ID construction with:

```python
inline_id = inline_spec.strategy_id if inline_spec is not None else ""
explicit_identity = top_level_id or inline_id
saved_strategy = (
    _strategy_registry().get_strategy(explicit_identity)
    if explicit_identity
    else None
)
```

When no strategy spec exists:

```python
if strategy_spec is None:
    strategy_spec = StrategySpec(
        strategy_id=f"adhoc_factor_{factor_name}",
        name=f"Factor baseline: {factor_name}",
        kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
        universe="",
        factors=[{"factor_id": factor_name}],
        portfolio={"top_n": requested_top_n},
        rebalance={"frequency": strategy_frequency},
    )
```

Set:

```python
effective_id = strategy_spec.strategy_id
```

Do not query Registry using the generated ad-hoc ID.

- [ ] **Step 4: Preserve explicit identity guards**

Keep these behaviors unchanged and covered:

```text
explicit top-level ID not found -> STRATEGY_NOT_FOUND
inline ID collides with a different saved spec -> SAVED_STRATEGY_SPEC_MISMATCH
saved code path -> GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED
```

- [ ] **Step 5: Run the Agent identity tests**

```bash
uv run pytest \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/agent/test_agent_backtest_config_spec_consistency.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/agent/test_adhoc_factor_strategy_identity.py \
        tests/unit/agent/test_backtest_pre_cache_identity.py
git commit -m "fix(agent): isolate adhoc factor strategy identity"
```

---

### Task 4: Replace Current Security Metadata with Point-in-Time Listing Semantics

**Files:**
- Create: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Create: `tests/unit/universe/test_pit_security_master.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Produces: `security_master_asof(stock_basic: pd.DataFrame, as_of_date: date) -> pd.DataFrame`.
- Current `name` is display-only and never used to infer historical ST state.
- Current `list_status` is never used to decide historical listing eligibility.
- Historical industry/theme selection without dated classification evidence returns typed `UNIVERSE_PIT_CLASSIFICATION_NOT_READY`.

- [ ] **Step 1: Write listing-window and current-name leakage tests**

Create `tests/unit/universe/test_pit_security_master.py`:

```python
from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.universe.pit_metadata import security_master_asof


def test_delisted_symbol_is_eligible_before_delist_date() -> None:
    current = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "Current Name",
                "list_status": "D",
                "list_date": "20000101",
                "delist_date": "20200110",
            }
        ]
    )

    observed = security_master_asof(current, date(2020, 1, 5))

    assert observed["symbol"].tolist() == ["000001.SZ"]
    assert observed["listed_as_of"].tolist() == [True]


def test_future_st_name_is_not_historical_st_evidence() -> None:
    current = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "ST Current Name",
                "list_status": "L",
                "list_date": "20000101",
                "delist_date": None,
            }
        ]
    )

    observed = security_master_asof(current, date(2010, 1, 5))

    assert observed["display_name"].tolist() == ["ST Current Name"]
    assert "st" not in observed.columns
```

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/unit/universe/test_pit_security_master.py -q
```

Expected: FAIL because the PIT metadata module does not exist.

- [ ] **Step 3: Implement listing-window normalization**

Create `src/qmt_agent_trader/universe/pit_metadata.py`:

```python
"""Point-in-time metadata used by universe resolution."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.data.integrity import require_unique_keys


def security_master_asof(
    stock_basic: pd.DataFrame,
    as_of_date: date,
) -> pd.DataFrame:
    if stock_basic.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "display_name",
                "list_date",
                "delist_date",
                "listed_as_of",
            ]
        )
    required = {"ts_code", "list_date"}
    missing = sorted(required.difference(stock_basic.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_SECURITY_MASTER_INVALID",
            message="stock_basic lacks listing-window fields",
            field="raw/tushare/stock_basic",
            details={"missing_columns": missing},
        )
    data = stock_basic.copy()
    require_unique_keys(
        data,
        keys=("ts_code",),
        code="DUPLICATE_UNIVERSE_SOURCE_KEY",
        field="raw/tushare/stock_basic",
    )
    data["symbol"] = data["ts_code"].astype(str)
    data["list_date"] = _required_date(
        data["list_date"],
        field="raw/tushare/stock_basic.list_date",
    )
    if "delist_date" in data.columns:
        data["delist_date"] = _optional_date(data["delist_date"])
    else:
        data["delist_date"] = pd.NaT
    data["display_name"] = (
        data["name"].astype("string")
        if "name" in data.columns
        else pd.Series(pd.NA, index=data.index, dtype="string")
    )
    data["listed_as_of"] = (
        data["list_date"].le(as_of_date)
        & (
            data["delist_date"].isna()
            | data["delist_date"].gt(as_of_date)
        )
    )
    return data[
        [
            "symbol",
            "display_name",
            "list_date",
            "delist_date",
            "listed_as_of",
        ]
    ]


def require_historical_classification_support(
    *,
    selection_mode: str,
    as_of_date: date,
    classification_frame: pd.DataFrame | None,
) -> None:
    if selection_mode not in {"industry", "theme"}:
        return
    required = {"symbol", "effective_from", "effective_to"}
    available = set(classification_frame.columns) if classification_frame is not None else set()
    if not required.issubset(available):
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_PIT_CLASSIFICATION_NOT_READY",
            message="historical industry/theme selection requires dated classification evidence",
            trade_date=as_of_date.isoformat(),
            field="classification_history",
            details={"selection_mode": selection_mode},
        )


def _required_date(values: pd.Series, *, field: str) -> pd.Series:
    parsed = pd.to_datetime(values.astype("string"), format="mixed", errors="coerce").dt.date
    if parsed.isna().any():
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_SECURITY_MASTER_INVALID",
            message="security master contains invalid listing dates",
            field=field,
            details={"invalid_row_count": int(parsed.isna().sum())},
        )
    return parsed


def _optional_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype("string"), format="mixed", errors="coerce").dt.date
```

- [ ] **Step 4: Use PIT listing metadata in `UniverseResolver`**

Import:

```python
from qmt_agent_trader.universe.pit_metadata import (
    require_historical_classification_support,
    security_master_asof,
)
```

In `_resolve_for_date()`:

```python
as_of = _parse_date(as_of_date)
stock_basic = security_master_asof(self._stock_basic(), as_of)
require_historical_classification_support(
    selection_mode=spec.selection.mode,
    as_of_date=as_of,
    classification_frame=None,
)
```

Update `_merge_stock_basic_columns()` to merge these columns:

```python
[
    "symbol",
    "display_name",
    "list_date",
    "delist_date",
    "listed_as_of",
]
```

- [ ] **Step 5: Remove current-name and current-status filtering**

Replace the listing/ST block in `_exclusion_reason()` with:

```python
if row.get("listed_as_of") is not True:
    return "not_listed_as_of"
list_date_raw = row.get("list_date")
if not _is_missing_scalar(list_date_raw):
    listed_days = (
        _parse_date(as_of_date)
        - _parse_date(str(list_date_raw))
    ).days
    if listed_days < filters.min_listed_days:
        return "listed_days_below_minimum"
if filters.exclude_st and bool(row["st"]):
    return "st"
```

Delete all use of:

```python
row.get("list_status")
"ST" in str(row.get("name", "")).upper()
```

- [ ] **Step 6: Add Resolver-level regressions**

Add tests proving:

```text
current list_status=D does not exclude a symbol before delist_date
current name containing ST does not exclude a historically non-ST symbol
industry selection raises UNIVERSE_PIT_CLASSIFICATION_NOT_READY without dated evidence
```

Use explicit `namechange` fixtures for the real ST interval test.

- [ ] **Step 7: Run focused universe tests**

```bash
uv run pytest \
  tests/unit/universe/test_pit_security_master.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/universe/test_universe_resolver_snapshot.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/universe/pit_metadata.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_pit_security_master.py \
        tests/unit/universe/test_resolver.py
git commit -m "fix(universe): use point in time listing metadata"
```

---

### Task 5: Resolve Universes from Exact Sessions and Bounded Lookbacks

**Files:**
- Modify: `src/qmt_agent_trader/data/trading_calendar.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Create: `tests/unit/universe/test_exact_session_resolution.py`
- Modify: `tests/unit/data/test_trading_calendar.py`

**Interfaces:**
- Produces: `latest_open_session_on_or_before(lake, as_of, exchanges=("SSE", "SZSE")) -> date`.
- `_load_recent_bars()` loads the exact effective open session only.
- `_avg_20d_metrics()` loads exactly 20 open sessions, not all history.
- A symbol without a bar on the effective session has no bar coverage for that session.

- [ ] **Step 1: Write the stale-bar regression**

Create `tests/unit/universe/test_exact_session_resolution.py`:

```python
from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.resolver import UniverseResolver


def test_previous_session_bar_is_not_current_session_coverage(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240103", "is_open": 1},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    _write_empty_trade_state_sources(lake)
    _write_stock_basic(lake, ["000001.SZ"])
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "all_stock",
            "name": "All stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )

    result = UniverseResolver(lake).build(
        spec,
        mode="snapshot",
        as_of_date="20240103",
    )

    assert result["status"] == "OK"
    assert result["symbols"] == []
```

Define the two fixture helpers in the same file with explicit schemas; do not import private test helpers across modules.

- [ ] **Step 2: Run the regression**

```bash
uv run pytest tests/unit/universe/test_exact_session_resolution.py -q
```

Expected: FAIL because the current resolver takes each symbol's latest historical row.

- [ ] **Step 3: Add calendar as-of resolution**

In `trading_calendar.py` add:

```python
def latest_open_session_on_or_before(
    lake: DataLake,
    *,
    as_of: str | date,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> date:
    boundary = (
        as_of
        if isinstance(as_of, date)
        else _parse_boundary(str(as_of))
    )
    states = _load_normalized_calendar_states(
        lake,
        exchanges=exchanges,
    )
    candidates = [
        day
        for day, is_open in states.items()
        if day <= boundary and is_open == 1
    ]
    if not candidates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_EMPTY",
            message="no open session exists on or before the requested date",
            field="trade_cal",
            details={"as_of": boundary.isoformat()},
        )
    return max(candidates)
```

- [ ] **Step 4: Load exact-session bars**

In `UniverseResolver._load_recent_bars()`:

```python
effective_date = latest_open_session_on_or_before(
    self.lake,
    as_of=as_of_date,
)
key = f"{effective_date:%Y%m%d}"
bars = load_daily_bars(
    self.lake,
    start=key,
    end=key,
    include_trade_state=True,
    asset_types=list(asset_types),
)
return bars[
    bars["trade_date"].eq(effective_date)
    & bars["asset_type"].isin(asset_types)
].reset_index(drop=True)
```

No `.groupby(...).tail(1)` remains in this path.

- [ ] **Step 5: Bound 20-session metric loading**

In `_avg_20d_metrics()`:

```python
effective_date = latest_open_session_on_or_before(
    self.lake,
    as_of=as_of_date,
)
window = load_session_window(
    self.lake,
    start=f"{effective_date:%Y%m%d}",
    end=f"{effective_date:%Y%m%d}",
    warmup_sessions=19,
)
start_key = f"{window.panel_start:%Y%m%d}"
end_key = f"{effective_date:%Y%m%d}"
```

Pass both `start=start_key` and `end=end_key` to raw readers. After concatenation, restrict rows to:

```python
allowed_dates = set((*window.warmup_dates, *window.expected_dates))
bars = bars[bars["trade_date"].isin(allowed_dates)]
```

Then aggregate all rows in that exact 20-session window. Do not read from the beginning of the dataset.

- [ ] **Step 6: Record effective resolution date**

Add to snapshot and rolling diagnostics:

```python
"effective_market_session": f"{effective_date:%Y%m%d}"
```

Ensure fingerprints include this resolved session through the existing resolved-universe payload.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/universe/test_universe_resolver_rolling.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/data/trading_calendar.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_exact_session_resolution.py \
        tests/unit/data/test_trading_calendar.py
git commit -m "fix(universe): resolve exact sessions with bounded history"
```

---

### Task 6: Implement Point-in-Time Index Membership

**Files:**
- Modify: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Create: `tests/unit/universe/test_index_membership_asof.py`

**Interfaces:**
- Produces: `index_weight_members_asof(frame, index_codes, as_of) -> list[str]`.
- Produces: `index_interval_members_asof(frame, index_codes, as_of) -> list[str]`.
- `index_weight`: latest snapshot date on or before `as_of` per index.
- `index_member`: `in_date <= as_of < out_date`, with null `out_date` treated as open-ended.

- [ ] **Step 1: Write the historical-union regressions**

Create `tests/unit/universe/test_index_membership_asof.py`:

```python
from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.universe.pit_metadata import (
    index_interval_members_asof,
    index_weight_members_asof,
)


def test_index_weight_uses_latest_snapshot_not_historical_union() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "OLD.SZ",
                "trade_date": "20240101",
            },
            {
                "index_code": "000300.SH",
                "con_code": "NEW.SZ",
                "trade_date": "20240201",
            },
        ]
    )

    observed = index_weight_members_asof(
        frame,
        ["000300.SH"],
        date(2024, 2, 15),
    )

    assert observed == ["NEW.SZ"]


def test_index_member_uses_effective_interval() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "OLD.SZ",
                "in_date": "20230101",
                "out_date": "20240131",
            },
            {
                "index_code": "000300.SH",
                "con_code": "NEW.SZ",
                "in_date": "20240201",
                "out_date": None,
            },
        ]
    )

    observed = index_interval_members_asof(
        frame,
        ["000300.SH"],
        date(2024, 2, 15),
    )

    assert observed == ["NEW.SZ"]
```

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/unit/universe/test_index_membership_asof.py -q
```

Expected: FAIL because the helpers do not exist.

- [ ] **Step 3: Implement index snapshot resolution**

Append to `pit_metadata.py`:

```python
def index_weight_members_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> list[str]:
    required = {"index_code", "con_code", "trade_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index_weight lacks required columns",
            field="raw/tushare/index_weight",
            details={"missing_columns": missing},
        )
    data = frame.copy()
    data["trade_date"] = _required_date(
        data["trade_date"],
        field="raw/tushare/index_weight.trade_date",
    )
    data = data[
        data["index_code"].astype(str).isin(index_codes)
        & data["trade_date"].le(as_of)
    ]
    members: list[str] = []
    for index_code, group in data.groupby("index_code", sort=True):
        snapshot_date = group["trade_date"].max()
        snapshot = group[group["trade_date"].eq(snapshot_date)]
        require_unique_keys(
            snapshot,
            keys=("index_code", "con_code", "trade_date"),
            code="DUPLICATE_UNIVERSE_SOURCE_KEY",
            field="raw/tushare/index_weight",
        )
        members.extend(snapshot["con_code"].astype(str).tolist())
    return sorted(dict.fromkeys(members))
```

- [ ] **Step 4: Implement interval membership resolution**

```python
def index_interval_members_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> list[str]:
    required = {"index_code", "con_code", "in_date", "out_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index_member lacks effective interval columns",
            field="raw/tushare/index_member",
            details={"missing_columns": missing},
        )
    data = frame.copy()
    data["in_date"] = _required_date(
        data["in_date"],
        field="raw/tushare/index_member.in_date",
    )
    data["out_date"] = _optional_date(data["out_date"])
    active = data[
        data["index_code"].astype(str).isin(index_codes)
        & data["in_date"].le(as_of)
        & (data["out_date"].isna() | data["out_date"].gt(as_of))
    ]
    require_unique_keys(
        active,
        keys=("index_code", "con_code"),
        code="DUPLICATE_UNIVERSE_SOURCE_KEY",
        field="raw/tushare/index_member",
    )
    return sorted(active["con_code"].astype(str).unique().tolist())
```

- [ ] **Step 5: Route Resolver through the helpers**

Replace `_index_constituents()` with:

```python
def _index_constituents(
    self,
    index_codes: list[str],
    as_of_date: str,
) -> list[str]:
    as_of = _parse_date(as_of_date)
    weight_path = self.lake.dataset_path("raw", "tushare/index_weight")
    if weight_path.exists():
        members = index_weight_members_asof(
            self.lake.read_parquet("raw", "tushare/index_weight"),
            index_codes,
            as_of,
        )
        if members:
            return [
                normalized
                for item in members
                if (normalized := normalize_symbol(item)) is not None
            ]
    member_path = self.lake.dataset_path("raw", "tushare/index_member")
    if member_path.exists():
        members = index_interval_members_asof(
            self.lake.read_parquet("raw", "tushare/index_member"),
            index_codes,
            as_of,
        )
        return [
            normalized
            for item in members
            if (normalized := normalize_symbol(item)) is not None
        ]
    return []
```

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/universe/pit_metadata.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_index_membership_asof.py
git commit -m "fix(universe): resolve index membership point in time"
```

---

### Task 7: Support ETF Opening State Without Stock ST Semantics

**Files:**
- Modify: `src/qmt_agent_trader/data/trade_state.py`
- Modify: `src/qmt_agent_trader/data/bars.py`
- Create: `tests/unit/data/test_etf_opening_trade_state.py`
- Modify: `tests/unit/data/test_asset_aware_trade_state.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Produces: `normalize_etf_opening_trade_state(bars, *, stk_limit) -> pd.DataFrame`.
- ETF rows use `stk_limit` for opening limits.
- ETF rows with a valid daily bar have `suspended=False` and `st=False`.
- Missing ETF daily bar is represented by absence of the row, not a stale carry-forward row.

- [ ] **Step 1: Write the ETF enrichment regression**

Create `tests/unit/data/test_etf_opening_trade_state.py`:

```python
from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.data.trade_state import normalize_etf_opening_trade_state


def test_etf_uses_limit_source_without_stock_st_state() -> None:
    bars = pd.DataFrame(
        [
            {
                "symbol": "510300.SH",
                "trade_date": date(2024, 1, 2),
                "asset_type": "etf",
                "open": 3.5,
                "high": 3.6,
                "low": 3.4,
                "close": 3.55,
                "volume": 1000.0,
                "amount": 3500.0,
                "turnover": pd.NA,
            }
        ]
    )
    limits = pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_date": "20240102",
                "up_limit": 3.85,
                "down_limit": 3.15,
            }
        ]
    )

    observed = normalize_etf_opening_trade_state(
        bars,
        stk_limit=limits,
    )

    assert observed["st"].tolist() == [False]
    assert observed["suspended"].tolist() == [False]
    assert observed["limit_up_at_open"].tolist() == [False]
    assert observed.attrs["trade_state_quality"]["asset_type"] == "etf"
```

- [ ] **Step 2: Run the regression**

```bash
uv run pytest tests/unit/data/test_etf_opening_trade_state.py -q
```

Expected: FAIL because the ETF normalizer does not exist.

- [ ] **Step 3: Extract shared opening-limit enrichment**

In `trade_state.py` add:

```python
def _apply_opening_limits(
    bars: pd.DataFrame,
    limits: pd.DataFrame,
) -> pd.DataFrame:
    result = bars.copy()
    result["symbol"] = result["symbol"].astype(str)
    result["trade_date"] = _coerce_dates(
        result["trade_date"],
        field="bars.trade_date",
    )
    require_unique_symbol_dates(
        result,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_SYMBOL_DATE_BAR",
        field="bars",
    )
    result = result.merge(
        limits,
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    missing = result["up_limit"].isna() | result["down_limit"].isna()
    if missing.any():
        raise BacktestDataIntegrityError(
            code="TRADE_STATE_PARTIAL_COVERAGE",
            message="limit source does not cover every executable bar",
            field="raw/tushare/stk_limit",
            symbols=tuple(
                sorted(result.loc[missing, "symbol"].astype(str).unique())
            ),
            details={"missing_key_count": int(missing.sum())},
        )
    opening_prices = pd.to_numeric(result["open"], errors="coerce")
    if opening_prices.isna().any():
        raise BacktestDataIntegrityError(
            code="INVALID_REQUIRED_PRICE",
            message="opening price is required for opening-limit state",
            field="open",
            details={"invalid_row_count": int(opening_prices.isna().sum())},
        )
    tolerance = 1e-6
    result["limit_up_at_open"] = (
        opening_prices >= result["up_limit"] - tolerance
    )
    result["limit_down_at_open"] = (
        opening_prices <= result["down_limit"] + tolerance
    )
    return result.drop(columns=["up_limit", "down_limit"])
```

Use this helper inside the stock normalizer.

- [ ] **Step 4: Implement the ETF normalizer**

```python
def normalize_etf_opening_trade_state(
    bars: pd.DataFrame,
    *,
    stk_limit: pd.DataFrame,
) -> pd.DataFrame:
    limits = _normalize_stock_limits(stk_limit)
    result = _apply_opening_limits(bars, limits)
    result["suspended"] = False
    result["st"] = False
    for column in OPENING_TRADE_STATE_COLUMNS:
        result[column] = result[column].astype(bool)
    result.attrs["column_quality"] = bars.attrs.get("column_quality", {})
    result.attrs["trade_state_quality"] = {
        "asset_type": "etf",
        "execution_time": "open",
        "suspended": {
            "source": "presence_of_valid_fund_daily_bar",
            "complete": True,
        },
        "st": {
            "source": "not_applicable_for_etf",
            "complete": True,
        },
        "limit_up_at_open": {
            "source": "raw/tushare/stk_limit",
            "complete": True,
        },
        "limit_down_at_open": {
            "source": "raw/tushare/stk_limit",
            "complete": True,
        },
    }
    return result
```

- [ ] **Step 5: Enrich stock and ETF partitions separately**

In `load_daily_bars()`:

```python
stock_bars = bars[bars["asset_type"].eq("stock")].copy()
etf_bars = bars[bars["asset_type"].eq("etf")].copy()
parts: list[pd.DataFrame] = []
if not stock_bars.empty:
    _require_trade_state_sources(lake, require_stock_metadata=True)
    parts.append(
        normalize_stock_opening_trade_state(
            stock_bars,
            suspend=suspend,
            stk_limit=stk_limit,
            namechange=namechange,
        )
    )
if not etf_bars.empty:
    _require_trade_state_sources(lake, require_stock_metadata=False)
    parts.append(
        normalize_etf_opening_trade_state(
            etf_bars,
            stk_limit=stk_limit,
        )
    )
bars = pd.concat(parts, ignore_index=True)
```

Change `_require_trade_state_sources()` so ETF-only calls require only `tushare/stk_limit`, while stock calls also require `suspend_d` and `namechange`.

Remove `UNSUPPORTED_ETF_TRADE_STATE_MODEL` from this path.

- [ ] **Step 6: Add mixed stock/ETF tests**

Update `tests/unit/data/test_asset_aware_trade_state.py` to assert:

```text
stock-only -> stock metadata semantics
ETF-only -> completed ETF state
mixed -> both assets returned with non-null canonical opening state
```

Update the UniverseResolver ETF fixture to prove an ETF with an exact as-of bar can be selected.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/unit/data/test_etf_opening_trade_state.py \
  tests/unit/data/test_asset_aware_trade_state.py \
  tests/unit/data/test_opening_trade_state.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/strategy/test_backtest_universe_resolution.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/data/trade_state.py \
        src/qmt_agent_trader/data/bars.py \
        tests/unit/data/test_etf_opening_trade_state.py \
        tests/unit/data/test_asset_aware_trade_state.py \
        tests/unit/universe/test_resolver.py
git commit -m "feat(data): support etf opening trade state"
```

---

### Task 8: Resolve Financial ASOF Revisions with Business Keys

**Files:**
- Modify: `src/qmt_agent_trader/data/fundamentals.py`
- Modify: `src/qmt_agent_trader/factors/input_panel.py`
- Create: `tests/unit/factors/test_financial_asof_resolution.py`
- Modify: `tests/unit/factors/test_asof_ambiguity.py`

**Interfaces:**
- Produces: `financial_field_asof_source(raw, *, field, source) -> pd.DataFrame`.
- Output columns: `symbol`, `visible_date`, `<field>`.
- Same-day financial revisions are ordered by `period_end`, `update_flag`, and `actual_announced_at`.
- Conflicting rows with identical complete business rank raise `AMBIGUOUS_FINANCIAL_REVISION`.
- Generic non-financial ASOF joins remain strict on visible keys.

- [ ] **Step 1: Write revision and ambiguity tests**

Create `tests/unit/factors/test_financial_asof_resolution.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.fundamentals import financial_field_asof_source


def test_same_day_financial_revision_uses_latest_period_and_update() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20230930",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "0",
                "roe": 8.0,
            },
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "1",
                "roe": 10.0,
            },
        ]
    )

    observed = financial_field_asof_source(
        raw,
        field="roe",
        source="tushare/fina_indicator",
    )

    assert observed.to_dict(orient="records") == [
        {
            "symbol": "000001.SZ",
            "visible_date": pd.Timestamp("2024-04-30").date(),
            "roe": 10.0,
        }
    ]


def test_identical_business_rank_with_conflicting_value_fails() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "1",
                "roe": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "end_date": "20231231",
                "ann_date": "20240430",
                "f_ann_date": "20240430",
                "update_flag": "1",
                "roe": 11.0,
            },
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        financial_field_asof_source(
            raw,
            field="roe",
            source="tushare/fina_indicator",
        )

    assert exc_info.value.code == "AMBIGUOUS_FINANCIAL_REVISION"
```

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/unit/factors/test_financial_asof_resolution.py -q
```

Expected: FAIL because `financial_field_asof_source` does not exist.

- [ ] **Step 3: Implement financial source reduction**

In `fundamentals.py` add:

```python
def financial_field_asof_source(
    raw: pd.DataFrame,
    *,
    field: str,
    source: str,
) -> pd.DataFrame:
    normalized = normalize_financial_statement(
        raw,
        statement_type=source.rsplit("/", 1)[-1],
        source=source,
    )
    required = {
        "symbol",
        "visible_date",
        "period_end",
        "actual_announced_at",
        "update_flag",
        field,
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise BacktestDataIntegrityError(
            code="INVALID_FINANCIAL_SOURCE",
            message="financial source lacks revision identity",
            field=source,
            details={"missing_columns": missing},
        )
    data = normalized.dropna(
        subset=["symbol", "visible_date", field]
    ).copy()
    data["update_rank"] = pd.to_numeric(
        data["update_flag"],
        errors="coerce",
    ).fillna(-1)
    rank_columns = [
        "symbol",
        "visible_date",
        "period_end",
        "update_rank",
        "actual_announced_at",
    ]
    duplicate_rank = data.duplicated(rank_columns, keep=False)
    if duplicate_rank.any():
        conflicts = (
            data.loc[duplicate_rank]
            .groupby(rank_columns, dropna=False)[field]
            .nunique(dropna=False)
        )
        if (conflicts > 1).any():
            raise BacktestDataIntegrityError(
                code="AMBIGUOUS_FINANCIAL_REVISION",
                message="financial revisions share an identical business rank",
                field=f"{source}:{field}",
                symbols=tuple(
                    sorted(
                        data.loc[duplicate_rank, "symbol"]
                        .astype(str)
                        .unique()
                    )
                ),
            )
        data = data.drop_duplicates(rank_columns)
    selected = (
        data.sort_values(
            [
                "symbol",
                "visible_date",
                "period_end",
                "update_rank",
                "actual_announced_at",
            ],
            kind="stable",
            na_position="first",
        )
        .groupby(["symbol", "visible_date"], as_index=False)
        .tail(1)
    )
    return selected[["symbol", "visible_date", field]].reset_index(drop=True)
```

Import `BacktestDataIntegrityError` in `fundamentals.py`.

- [ ] **Step 4: Route financial sources before generic ASOF validation**

In `input_panel.py`, import:

```python
from qmt_agent_trader.data.fundamentals import financial_field_asof_source
```

In `_join_asof_snapshot_field()` after reading `raw`:

```python
if source.api_name in {
    "income",
    "balancesheet",
    "cashflow",
    "fina_indicator",
}:
    data = financial_field_asof_source(
        raw,
        field=source_field,
        source=source.raw_dataset_name,
    ).rename(columns={source_field: field})
    return _join_symbol_asof(panel, data, field)
```

Keep `require_unique_keys(symbol, visible_date)` in `_join_symbol_asof()`; the financial reducer now creates that canonical identity.

- [ ] **Step 5: Preserve generic ambiguity behavior**

Keep existing `DUPLICATE_ASOF_VISIBLE_KEY` tests for non-financial source data. Do not weaken the generic join.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/unit/factors/test_financial_asof_resolution.py \
  tests/unit/factors/test_asof_ambiguity.py \
  tests/unit/factors/test_input_panel.py \
  tests/unit/test_fundamentals_pit.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/data/fundamentals.py \
        src/qmt_agent_trader/factors/input_panel.py \
        tests/unit/factors/test_financial_asof_resolution.py \
        tests/unit/factors/test_asof_ambiguity.py
git commit -m "fix(factors): resolve financial revisions by business key"
```

---

### Task 9: Strengthen Provenance Fingerprints and Finalize Verification

**Files:**
- Modify: `src/qmt_agent_trader/persistence/provenance.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/persistence/test_provenance_content_hash.py`
- Modify: `tests/unit/agent/test_backtest_cache_provenance.py`
- Modify: `docs/backtest/factor-rank-adapter.md`

**Interfaces:**
- Small control files and code files are content-hashed.
- Parquet datasets use a deterministic tree digest that includes a bounded content sample from every file, in addition to path and size.
- Cache engine semantic version changes to `2026-07-factor-universe-pit-v2`.

- [ ] **Step 1: Write the same-size, preserved-mtime regression**

Create `tests/unit/persistence/test_provenance_content_hash.py`:

```python
from __future__ import annotations

import os

from qmt_agent_trader.persistence.provenance import fingerprint_path_tree


def test_same_size_replacement_with_preserved_mtime_changes_fingerprint(
    tmp_path,
) -> None:
    path = tmp_path / "registry.json"
    path.write_bytes(b"AAAA")
    stat = path.stat()
    first = fingerprint_path_tree(path)

    path.write_bytes(b"BBBB")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = fingerprint_path_tree(path)

    assert first != second
```

- [ ] **Step 2: Run the regression**

```bash
uv run pytest tests/unit/persistence/test_provenance_content_hash.py -q
```

Expected: FAIL because the existing digest uses only path, size, and mtime.

- [ ] **Step 3: Hash content deterministically**

Replace the per-file update in `provenance.py` with:

```python
_CONTENT_CHUNK_BYTES = 1024 * 1024


def _content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CONTENT_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _update_file(
    digest: "hashlib._Hash",
    item: Path,
    *,
    root: Path,
) -> None:
    stat = item.stat()
    digest.update(
        (
            f"{item.relative_to(root).as_posix()}\0"
            f"{stat.st_size}\0"
            f"{_content_digest(item)}\n"
        ).encode("utf-8")
    )
```

Do not include `mtime_ns` in the semantic digest. File content and relative path define the cache identity.

- [ ] **Step 4: Update the engine semantic version**

In `strategy_tools.py`:

```python
BACKTEST_ENGINE_SEMANTIC_VERSION = "2026-07-factor-universe-pit-v2"
```

This invalidates results produced under the stale-universe and incorrect-factor semantics.

- [ ] **Step 5: Add cache invalidation assertions**

Extend `test_backtest_cache_provenance.py` so changes to each of these alter the cache key:

```text
daily_basic turnover_rate content
stock_basic list_date/delist_date content
index_weight or index_member content
fina_indicator revision content
factor implementation file content
strategy code content
```

Use tiny files in the test; do not create large synthetic Parquet datasets.

- [ ] **Step 6: Update documentation**

Update `docs/backtest/factor-rank-adapter.md` with these exact contracts:

```markdown
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

## ETF opening state

ETF opening limit state uses `tushare/stk_limit`. ETF ST state is not applicable, and
presence of a valid exact-session `fund_daily` row is the evidence that the row is
tradable rather than suspended.

## Financial revisions

Financial ASOF fields are reduced by symbol, visible date, report period, update flag,
and actual announcement date before the generic ASOF join. Identical business ranks
with conflicting values fail closed.
```

- [ ] **Step 7: Run all new focused tests**

```bash
uv run pytest \
  tests/unit/factors/test_price_volume_group_boundaries.py \
  tests/unit/factors/test_canonical_turnover_source.py \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/universe/test_pit_security_master.py \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/data/test_etf_opening_trade_state.py \
  tests/unit/factors/test_financial_asof_resolution.py \
  tests/unit/persistence/test_provenance_content_hash.py -q
```

Expected: PASS.

- [ ] **Step 8: Run affected subsystem suites**

```bash
uv run pytest \
  tests/unit/backtest \
  tests/unit/data \
  tests/unit/factors \
  tests/unit/strategy \
  tests/unit/universe \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py -q
```

Expected: PASS.

- [ ] **Step 9: Inspect prohibited historical shortcuts**

```bash
rg -n 'list_status|"ST" in|drop_duplicates\(.*keep=|groupby\(.*tail\(1\)|row_number\(' \
  src/qmt_agent_trader/universe \
  src/qmt_agent_trader/factors \
  src/qmt_agent_trader/data
```

Every match must satisfy one of these conditions:

- display-only metadata;
- deduplication after a complete business-key equality check;
- latest selection after an explicit PIT ordering key;
- test-only code.

No historical eligibility decision may use current `list_status` or current name text.

- [ ] **Step 10: Inspect unbounded universe reads**

```bash
rg -n 'read_parquet_filtered\(' src/qmt_agent_trader/universe/resolver.py
```

Confirm:

```text
exact-session bars pass start and end
20-session metrics pass a bounded start and end
index sources are reduced by PIT semantics
```

- [ ] **Step 11: Run repository gates**

```bash
make check
```

Expected: exit code `0`.

- [ ] **Step 12: Commit**

```bash
git add src/qmt_agent_trader/persistence/provenance.py \
        src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/persistence/test_provenance_content_hash.py \
        tests/unit/agent/test_backtest_cache_provenance.py \
        docs/backtest/factor-rank-adapter.md
git commit -m "docs(backtest): finalize factor and universe correctness"
```

---

## Final Acceptance Checklist

### Factor correctness

- [ ] `volatility_20d` never crosses a symbol boundary.
- [ ] `turnover` comes from `daily_basic.turnover_rate`.
- [ ] An all-null placeholder cannot suppress field-source discovery.
- [ ] `turnover_20d` produces values after the declared 20-session lookback.
- [ ] Financial revisions use business-key ordering rather than storage order.
- [ ] Truly ambiguous financial revisions fail closed.

### Strategy identity

- [ ] Pure `factor_name` requests never read a saved strategy implicitly.
- [ ] Explicit strategy IDs still enforce Registry identity and generated-code guards.
- [ ] Ad-hoc factor IDs are deterministic and cache-stable.

### Historical universe correctness

- [ ] Current company name is never used as historical ST evidence.
- [ ] Current `list_status` is never used as historical listing evidence.
- [ ] `list_date <= as_of < delist_date` defines listing eligibility.
- [ ] Historical industry/theme selection blocks without dated classifications.
- [ ] Snapshot resolution uses an exact effective open session.
- [ ] A stale previous-session bar is not current bar coverage.
- [ ] 20-session liquidity metrics use a bounded 20-session window.
- [ ] Index-weight selection uses the latest snapshot, not historical union.
- [ ] Index-member selection uses effective intervals.

### ETF correctness

- [ ] ETF rows use `stk_limit` opening prices.
- [ ] ETF rows do not use stock name-change/ST semantics.
- [ ] ETF rows require an exact-session `fund_daily` bar.
- [ ] Mixed stock/ETF panels contain complete non-null opening-state fields.

### Provenance and safety

- [ ] Same-size content replacement changes the provenance fingerprint.
- [ ] Cache engine semantic version is bumped.
- [ ] Relevant data, Registry, factor, strategy, index, and financial changes invalidate cache.
- [ ] Integrity errors create no completed report or successful cache entry.
- [ ] Unexpected exceptions propagate.
- [ ] `research_only=True` and `live_trading_allowed=False` remain unchanged.
- [ ] Focused tests pass.
- [ ] Affected subsystem suites pass.
- [ ] `make check` passes.

## Explicitly Out of Scope

- Historical extreme-drawdown reproduction.
- A new full DataLake-to-Agent end-to-end suite.
- GitHub Actions creation or modification.
- Generated strategy Python process isolation.
- A general historical industry-classification ingestion system; this plan fails closed until dated evidence exists.

## Expected Merge Decision

Keep `REQUEST_CHANGES` until every acceptance item and local verification command passes. After implementation, perform one final static review of the branch before merging.
