# Factor-Rank Backtest Final Review Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining correctness gaps on `codex/factor-rank-backtest-correctness` so raw-data ambiguity is never normalized away, saved strategy identity is immutable, strategy and universe frequencies have unambiguous meanings, factor warm-up is correct, and all completed evidence is available through the canonical schema.

**Architecture:** Keep the existing factor-rank runner and typed integrity-error hierarchy. Move fail-closed validation to the earliest layer that still has the original evidence: raw normalization validates duplicate identities, trade-state loading validates source availability, the Agent boundary resolves strategy identity and frequency semantics before universe work, and the adapter loads a calendar-derived factor warm-up window before constructing the runner. Canonical report fields are sourced from typed result models rather than legacy payload-only fields.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, dataclasses, pytest, existing `DataLake`, `FactorRegistry`, `StrategyRegistry`, `UniverseResolver`, factor-rank runner, Ruff, mypy, and `uv`.

## Branch and Plan Location

- Repository: `huangyicong0769/qmt-agent-trader`
- Target branch: `codex/factor-rank-backtest-correctness`
- Continue from the current branch head; do not restart from `main`.
- Save this plan in the repository as:
  `docs/superpowers/plans/2026-07-14-factor-rank-backtest-final-review-repairs.md`
- Keep each task in a separate focused commit.

## Global Constraints

- Fail closed on ambiguous data; do not silently choose one duplicate row.
- Do not infer a missing trade-state source as `False`.
- `StrategySpec` remains authoritative whenever a strategy spec exists.
- A saved Registry strategy cannot be replaced by an inline spec carrying the same identity.
- `rebalance_frequency` means strategy rebalance frequency only.
- A separately named `universe_rebalance_frequency` may control rolling-universe snapshot cadence.
- Factor calculations must include their declared warm-up history.
- Warm-up bars are factor inputs only; they must not create pre-start trades, equity points, or metrics.
- Only typed `BacktestIntegrityError` subclasses are converted to structured `ERROR` at the outer Agent boundary.
- Unexpected programming exceptions propagate.
- Integrity failures create no completed report and no successful cache entry.
- Preserve `research_only=True` and `live_trading_allowed=False`.
- Do not add runtime dependencies.
- Follow TDD for every task: failing test, verify failure, minimal implementation, verify pass, commit.
- Do not add or modify GitHub Actions.
- Do not reproduce the historical extreme-drawdown run.
- Do not build a new full DataLake-to-Agent end-to-end fixture. Small component tests that exercise one production boundary are required.

---

## File Responsibility Map

### New files

- `src/qmt_agent_trader/data/integrity.py`
  Shared symbol-date uniqueness validation for raw/normalized market and exact-source frames.

- `tests/unit/data/test_data_integrity.py`
  Unit tests for the shared uniqueness validator.

- `tests/unit/data/test_trade_state_evidence.py`
  Component tests for missing and partial trade-state sources.

- `tests/unit/strategy/test_backtest_warmup.py`
  Adapter-level factor warm-up tests.

- `tests/unit/strategy/test_saved_strategy_identity.py`
  Registry identity and inline-spec conflict tests.

### Existing files to modify

- `src/qmt_agent_trader/data/bars.py`
  Remove silent daily-bar deduplication and require trade-state source evidence.

- `src/qmt_agent_trader/factors/input_panel.py`
  Reject duplicate exact-source rows instead of keeping the last row.

- `src/qmt_agent_trader/data/trading_calendar.py`
  Resolve prior open sessions for factor warm-up.

- `src/qmt_agent_trader/agent/tools/strategy_tools.py`
  Resolve saved strategy identity, separate strategy and universe frequencies, and block mismatches before universe resolution/cache lookup.

- `src/qmt_agent_trader/strategy/execution_adapter.py`
  Validate identity, derive warm-up requirements, load warm-up panel, and expose warm-up metadata.

- `src/qmt_agent_trader/backtest/research_runner.py`
  Accept pre-start factor-input bars while valuing only expected trade dates.

- `src/qmt_agent_trader/backtest/research_models.py`
  Place signal-availability counts in canonical `ResearchDataQuality`.

- `src/qmt_agent_trader/universe/resolver.py`
  Add deterministic symbol tie-breaking to ranking.

- `docs/backtest/factor-rank-adapter.md`
  Document the final contracts.

---

# Task 1: Reject Duplicates Before Normalization or Exact Joins

**Files:**
- Create: `src/qmt_agent_trader/data/integrity.py`
- Create: `tests/unit/data/test_data_integrity.py`
- Modify: `src/qmt_agent_trader/data/bars.py`
- Modify: `src/qmt_agent_trader/factors/input_panel.py`
- Modify: `tests/unit/factors/test_input_panel.py`
- Modify: `tests/unit/backtest/test_duplicate_inputs.py`

**Interfaces:**
- Produces:
  - `require_unique_keys(frame, *, keys, code, field) -> None`
  - `require_unique_symbol_dates(frame, *, symbol_column, date_column, code, field) -> None`
- Both identical and conflicting duplicates raise `BacktestDataIntegrityError`.
- No production data path may use `drop_duplicates(..., keep="last")` to resolve market or exact-source identity conflicts.

- [ ] **Step 1: Write the shared-validator tests**

Create `tests/unit/data/test_data_integrity.py`:

```python
from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.integrity import require_unique_symbol_dates


@pytest.mark.parametrize("second_close", [10.0, 11.0])
def test_symbol_date_duplicates_are_always_rejected(second_close: float) -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "close": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "close": second_close,
            },
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        require_unique_symbol_dates(
            frame,
            symbol_column="ts_code",
            date_column="trade_date",
            code="DUPLICATE_SYMBOL_DATE_BAR",
            field="raw/tushare/daily",
        )

    assert exc_info.value.code == "DUPLICATE_SYMBOL_DATE_BAR"
    assert exc_info.value.symbols == ("000001.SZ",)
    assert exc_info.value.details["duplicate_key_count"] == 1


def test_unique_symbol_dates_pass() -> None:
    frame = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": date(2024, 1, 2)},
            {"symbol": "A", "trade_date": date(2024, 1, 3)},
        ]
    )

    require_unique_symbol_dates(
        frame,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_SYMBOL_DATE_BAR",
        field="bars",
    )
```

- [ ] **Step 2: Run the validator tests**

```bash
uv run pytest tests/unit/data/test_data_integrity.py -q
```

Expected: FAIL because `qmt_agent_trader.data.integrity` does not exist.

- [ ] **Step 3: Implement the shared validator**

Create `src/qmt_agent_trader/data/integrity.py`:

```python
"""Fail-closed validation for tabular data identities."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError


def require_unique_keys(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    code: str,
    field: str,
) -> None:
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise BacktestDataIntegrityError(
            code="INVALID_IDENTITY_FRAME",
            message="identity validation requires missing columns",
            field=field,
            details={"missing_columns": missing},
        )
    duplicate_mask = frame.duplicated(list(keys), keep=False)
    if not duplicate_mask.any():
        return
    duplicate_keys = (
        frame.loc[duplicate_mask, list(keys)]
        .drop_duplicates()
        .sort_values(list(keys), kind="stable")
    )
    symbols: tuple[str, ...] = ()
    for symbol_column in ("symbol", "ts_code", "con_code"):
        if symbol_column in duplicate_keys.columns:
            symbols = tuple(
                sorted(duplicate_keys[symbol_column].astype(str).unique().tolist())
            )
            break
    sample = [
        {key: _jsonable_identity_value(getattr(row, key)) for key in keys}
        for row in duplicate_keys.head(20).itertuples(index=False)
    ]
    raise BacktestDataIntegrityError(
        code=code,
        message="identity columns must be unique",
        symbols=symbols,
        field=field,
        details={
            "identity_columns": list(keys),
            "duplicate_key_count": len(duplicate_keys),
            "sample": sample,
        },
    )


def require_unique_symbol_dates(
    frame: pd.DataFrame,
    *,
    symbol_column: str,
    date_column: str,
    code: str,
    field: str,
) -> None:
    require_unique_keys(
        frame,
        keys=(symbol_column, date_column),
        code=code,
        field=field,
    )


def _jsonable_identity_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return str(value)
```

- [ ] **Step 4: Validate raw daily rows before renaming/deduplicating**

In `normalize_tushare_daily()`:

```python
from qmt_agent_trader.data.integrity import require_unique_symbol_dates
```

After checking required raw columns and before renaming:

```python
require_unique_symbol_dates(
    data,
    symbol_column="ts_code" if "ts_code" in data.columns else "symbol",
    date_column="trade_date",
    code="DUPLICATE_SYMBOL_DATE_BAR",
    field="raw_daily_bars",
)
```

Replace:

```python
data[CANONICAL_BAR_COLUMNS]
.drop_duplicates(["symbol", "trade_date"], keep="last")
.sort_values(...)
```

with:

```python
data[CANONICAL_BAR_COLUMNS]
.sort_values(["symbol", "trade_date"], kind="stable")
.reset_index(drop=True)
```

- [ ] **Step 5: Add a production-path bar normalization test**

Append to `tests/unit/data/test_data_integrity.py`:

```python
from qmt_agent_trader.data.bars import normalize_tushare_daily


def test_daily_normalization_does_not_hide_duplicates() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.1,
                "high": 10.6,
                "low": 9.6,
                "close": 10.2,
            },
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        normalize_tushare_daily(raw)

    assert exc_info.value.code == "DUPLICATE_SYMBOL_DATE_BAR"
```

- [ ] **Step 6: Reject duplicate exact-source rows before joins**

In `_join_exact_field()` after constructing `data["trade_date"]` and before selecting/merging:

```python
require_unique_symbol_dates(
    data,
    symbol_column="symbol",
    date_column="trade_date",
    code="DUPLICATE_EXACT_FACTOR_INPUT",
    field=f"raw/{source.raw_dataset_name}:{field}",
)
data = data[["symbol", "trade_date", field]].dropna(
    subset=["symbol", "trade_date"]
)
```

Delete `.drop_duplicates(["symbol", "trade_date"], keep="last")`.

- [ ] **Step 7: Add the exact-source component regression**

Append to `tests/unit/factors/test_input_panel.py` using the existing `DataLake` fixture style:

```python
def test_exact_factor_source_duplicates_fail_closed(tmp_path) -> None:
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
                    "amount": 1_000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "pb": 1.0},
                {"ts_code": "000001.SZ", "trade_date": "20240102", "pb": 1.2},
            ]
        ),
        "raw",
        "tushare/daily_basic",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        build_target_frequency_panel(
            lake,
            target_frequency=Frequency.DAILY,
            target_start="20240102",
            target_end="20240102",
            required_fields=["symbol", "trade_date", "open", "close", "pb"],
            symbols=["000001.SZ"],
        )

    assert exc_info.value.code == "DUPLICATE_EXACT_FACTOR_INPUT"
```

- [ ] **Step 8: Reuse the shared validator in the runner**

Replace the local body of `_require_unique_symbol_trade_dates()` with a call to `require_unique_symbol_dates()`, or remove the local function and import the shared helper. Preserve the runner error codes:

```python
require_unique_symbol_dates(
    data,
    symbol_column="symbol",
    date_column="trade_date",
    code="DUPLICATE_SYMBOL_DATE_BAR",
    field="bars",
)
```

and:

```python
require_unique_symbol_dates(
    self.factor_frame,
    symbol_column="symbol",
    date_column="trade_date",
    code="DUPLICATE_FACTOR_SYMBOL_DATE",
    field="factor_frame",
)
```

- [ ] **Step 9: Run focused tests**

```bash
uv run pytest \
  tests/unit/data/test_data_integrity.py \
  tests/unit/factors/test_input_panel.py \
  tests/unit/backtest/test_duplicate_inputs.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/qmt_agent_trader/data/integrity.py \
        src/qmt_agent_trader/data/bars.py \
        src/qmt_agent_trader/factors/input_panel.py \
        src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/data/test_data_integrity.py \
        tests/unit/factors/test_input_panel.py \
        tests/unit/backtest/test_duplicate_inputs.py
git commit -m "fix(data): reject duplicate market and factor inputs"
```

---

# Task 2: Give Strategy and Universe Rebalance Frequencies Separate Meanings

**Files:**
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Modify: `tests/unit/agent/test_agent_backtest_config_spec_consistency.py`
- Modify: `tests/unit/strategy/test_backtest_rolling_universe.py`
- Modify: `tests/unit/agent/test_strategy_workflow.py`

**Interfaces:**
- `rebalance_frequency` is a strategy-semantic input.
- `universe_rebalance_frequency` controls rolling-universe resolution only.
- When a spec exists, conflicting `rebalance_frequency` returns `CONFIG_SPEC_MISMATCH`.
- For a factor-only temporary strategy, `rebalance_frequency` is written into the temporary `StrategySpec`.
- When `universe_rebalance_frequency` is absent, rolling-universe cadence defaults to the authoritative strategy frequency.

- [ ] **Step 1: Add the spec mismatch regression**

Append to `tests/unit/agent/test_agent_backtest_config_spec_consistency.py`:

```python
def test_strategy_rebalance_frequency_mismatch_blocks_before_universe_access(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)
    monkeypatch.setattr(
        strategy_tools,
        "_resolve_backtest_universe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatch must block before universe resolution")
        ),
    )

    result = strategy_tools._run_backtest(
        {
            "strategy_spec": _spec().model_dump(mode="json"),
            "rebalance_frequency": "daily",
            "start_date": "20240101",
            "end_date": "20240630",
        },
        ToolContext(run_id="frequency-mismatch"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "CONFIG_SPEC_MISMATCH"
    assert result["unsupported_fields"] == ["config.rebalance_frequency"]
```

- [ ] **Step 2: Add the factor-only temporary-spec regression**

Append to `tests/unit/agent/test_strategy_workflow.py`:

```python
def test_factor_only_backtest_builds_temporary_weekly_spec(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_strategy_backtest(_lake, _registry, config, *, reports_dir):
        captured["config"] = config
        return StrategyBacktestResult(
            run_id="research_fixture",
            strategy_id=config.strategy_id,
            strategy_version=config.strategy_spec.version,
            status="BLOCKED",
            reason="fixture",
        )

    # Reuse the file's existing lake, factor registry, and explicit-symbol setup.
    monkeypatch.setattr(strategy_tools, "run_strategy_backtest", fake_run_strategy_backtest)

    strategy_tools._run_backtest(
        {
            "factor_name": "momentum_20d",
            "rebalance_frequency": "weekly",
            "symbols": ["000001.SZ"],
            "start_date": "20240101",
            "end_date": "20240331",
        },
        ToolContext(run_id="temporary-weekly"),
    )

    assert captured["config"].strategy_spec.rebalance.frequency == "weekly"
    assert captured["config"].rebalance_frequency == "weekly"
```

Adapt setup imports and factor-registry monkeypatches to the existing test module rather than creating duplicate fixtures.

- [ ] **Step 3: Add the rolling-universe cadence regression**

In `tests/unit/strategy/test_backtest_rolling_universe.py`, capture the frequency passed to `UniverseResolver.build()`:

```python
def test_universe_frequency_is_explicitly_separate_from_strategy_frequency(
    monkeypatch,
    tmp_path,
) -> None:
    observed = {}

    def fake_build(self, spec, **kwargs):
        observed["frequency"] = kwargs["rebalance_frequency"]
        return {
            "status": "OK",
            "rolling_symbols": {"20240102": ["000001.SZ"]},
            "metadata": {"resolve_dates": ["20240102"], "empty_dates": []},
        }

    monkeypatch.setattr(UniverseResolver, "build", fake_build)

    result = strategy_tools._run_backtest(
        {
            "strategy_spec": weekly_strategy_spec(),
            "universe_spec": rolling_universe_spec(),
            "universe_mode": "rolling",
            "universe_rebalance_frequency": "monthly",
            "start_date": "20240101",
            "end_date": "20240331",
        },
        ToolContext(run_id="separate-universe-frequency"),
    )

    assert observed["frequency"] == "monthly"
```

The returned result may be blocked later by the deliberately minimal fixture; assert only the captured production argument.

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest \
  tests/unit/agent/test_agent_backtest_config_spec_consistency.py \
  tests/unit/strategy/test_backtest_rolling_universe.py \
  tests/unit/agent/test_strategy_workflow.py -q
```

Expected: at least the mismatch and temporary-spec tests FAIL.

- [ ] **Step 5: Parse strategy frequency before semantic config construction**

In `_run_backtest()`:

```python
requested_strategy_frequency = input_data.get("rebalance_frequency")
if requested_strategy_frequency is not None:
    requested_strategy_frequency = str(requested_strategy_frequency)
    if requested_strategy_frequency not in {"daily", "weekly", "monthly"}:
        return {
            "status": "INVALID_REQUEST",
            "reason": "UNSUPPORTED_REBALANCE_FREQUENCY",
            "allowed_values": ["daily", "weekly", "monthly"],
        }
```

When a `strategy_spec` already exists, set the semantic transport field to the requested value when provided so the existing validator detects mismatch:

```python
rebalance_frequency=cast(
    Literal["daily", "weekly", "monthly"],
    requested_strategy_frequency or strategy_spec.rebalance.frequency,
),
```

- [ ] **Step 6: Build factor-only temporary specs with the requested frequency**

Before constructing the temporary `StrategySpec`:

```python
temporary_frequency = cast(
    Literal["daily", "weekly", "monthly"],
    requested_strategy_frequency or "daily",
)
```

Then:

```python
strategy_spec = StrategySpec(
    strategy_id=strategy_id or f"factor_{factor_name}",
    name=f"Factor baseline: {factor_name}",
    kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
    factors=[{"factor_id": factor_name}],
    portfolio={"top_n": top_n},
    rebalance={"frequency": temporary_frequency},
)
```

- [ ] **Step 7: Separate rolling-universe frequency**

Add to the tool input schema:

```python
"universe_rebalance_frequency": {
    "type": "string",
    "enum": ["daily", "weekly", "monthly"],
},
```

In `_resolve_backtest_universe()` replace the current use of `input_data["rebalance_frequency"]` with:

```python
strategy_frequency = (
    strategy_spec.rebalance.frequency
    if strategy_spec is not None
    else spec.rebalance_frequency
)
universe_frequency = str(
    input_data.get("universe_rebalance_frequency")
    or strategy_frequency
)
```

Pass `universe_frequency` to `resolver.build()`.

- [ ] **Step 8: Run focused tests**

```bash
uv run pytest \
  tests/unit/agent/test_agent_backtest_config_spec_consistency.py \
  tests/unit/strategy/test_backtest_rolling_universe.py \
  tests/unit/agent/test_strategy_workflow.py \
  tests/unit/strategy/test_backtest_config_spec_consistency.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/agent/test_agent_backtest_config_spec_consistency.py \
        tests/unit/strategy/test_backtest_rolling_universe.py \
        tests/unit/agent/test_strategy_workflow.py
git commit -m "fix(strategy): separate strategy and universe frequencies"
```

---

# Task 3: Lock Saved Strategy Identity to the Registry Spec

**Files:**
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/strategy/test_saved_strategy_identity.py`
- Modify: `tests/unit/agent/test_saved_generated_strategy_backtest_guard.py`

**Interfaces:**
- Produces:
  - `strategy_spec_fingerprint(spec: StrategySpec) -> str`
  - `validate_strategy_identity(config, saved_strategy) -> tuple[AdapterCapabilityIssue, ...]`
- `config.strategy_id != spec.strategy_id` returns `CONFIG_SPEC_MISMATCH`.
- Inline spec differing from a saved strategy with the same ID returns `SAVED_STRATEGY_SPEC_MISMATCH`.
- When a saved strategy exists and matches, the Registry copy is the authoritative spec.

- [ ] **Step 1: Write identity tests**

Create `tests/unit/strategy/test_saved_strategy_identity.py`:

```python
from pathlib import Path

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def _spec(strategy_id: str = "saved_value", *, top_n: int = 10) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": strategy_id,
            "name": "Saved value",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "pb_rank", "ascending": True}],
            "portfolio": {"top_n": top_n},
            "rebalance": {"frequency": "weekly"},
            "execution": {"execution_delay_days": 1},
        }
    )


def _save(registry: StrategyRegistry, spec: StrategySpec) -> None:
    registry.save_strategy(
        SavedStrategy(
            strategy_id=spec.strategy_id,
            spec=spec,
            status="saved",
            code_path=None,
        )
    )


def test_config_strategy_id_must_equal_inline_spec_id(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    config = StrategyBacktestConfig(
        strategy_id="different_id",
        strategy_spec=_spec(),
        factor_name="pb_rank",
        start_date="20240101",
        end_date="20240331",
        top_n=10,
        rebalance_frequency="weekly",
        lower_is_better=True,
    )

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        config,
        reports_dir=Path(tmp_path / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "CONFIG_SPEC_MISMATCH"
    assert result.unsupported_fields == ["config.strategy_id"]


def test_inline_spec_cannot_replace_saved_registry_spec(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    registry = StrategyRegistry(tmp_path / "strategies")
    _save(registry, _spec(top_n=10))
    config = StrategyBacktestConfig(
        strategy_id="saved_value",
        strategy_spec=_spec(top_n=20),
        factor_name="pb_rank",
        start_date="20240101",
        end_date="20240331",
        top_n=20,
        rebalance_frequency="weekly",
        lower_is_better=True,
    )

    result = run_strategy_backtest(
        lake,
        registry,
        config,
        reports_dir=Path(tmp_path / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "SAVED_STRATEGY_SPEC_MISMATCH"
```

Adjust `_save()` to the exact `StrategyRegistry.save_strategy()` signature already present in the repository; preserve the assertions and identities.

- [ ] **Step 2: Run the identity tests**

```bash
uv run pytest tests/unit/strategy/test_saved_strategy_identity.py -q
```

Expected: FAIL because inline spec currently overrides the Registry spec.

- [ ] **Step 3: Add deterministic spec fingerprinting**

In `execution_adapter.py`:

```python
import hashlib


def strategy_spec_fingerprint(spec: StrategySpec) -> str:
    encoded = json.dumps(
        spec.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Validate identity before capability checks**

Immediately after loading `saved_strategy`:

```python
inline_spec = config.strategy_spec
if inline_spec is not None and config.strategy_id != inline_spec.strategy_id:
    issue = AdapterCapabilityIssue(
        field="config.strategy_id",
        observed=config.strategy_id,
        supported=inline_spec.strategy_id,
        message="config strategy_id must equal StrategySpec.strategy_id",
    )
    return StrategyBacktestResult(
        run_id=run_id,
        strategy_id=config.strategy_id,
        strategy_version=inline_spec.version,
        status="BLOCKED",
        reason="CONFIG_SPEC_MISMATCH",
        unsupported_fields=[issue.field],
        capability_issues=[asdict(issue)],
        research_only=True,
        live_trading_allowed=False,
    )

if saved_strategy is not None and inline_spec is not None:
    if strategy_spec_fingerprint(saved_strategy.spec) != strategy_spec_fingerprint(inline_spec):
        issue = AdapterCapabilityIssue(
            field="strategy_spec",
            observed=strategy_spec_fingerprint(inline_spec),
            supported=strategy_spec_fingerprint(saved_strategy.spec),
            message="inline StrategySpec differs from the saved Registry strategy",
        )
        return StrategyBacktestResult(
            run_id=run_id,
            strategy_id=config.strategy_id,
            strategy_version=saved_strategy.spec.version,
            status="BLOCKED",
            reason="SAVED_STRATEGY_SPEC_MISMATCH",
            unsupported_fields=[issue.field],
            capability_issues=[asdict(issue)],
            research_only=True,
            live_trading_allowed=False,
        )
```

Then resolve:

```python
spec = saved_strategy.spec if saved_strategy is not None else inline_spec
```

- [ ] **Step 5: Resolve the Registry strategy even when inline spec is present**

In `_run_backtest()`:

```python
if strategy_id:
    saved_strategy = _strategy_registry().get_strategy(str(strategy_id))
```

Do this regardless of whether `strategy_spec` was supplied. Then:

```python
if saved_strategy is not None and strategy_spec is not None:
    if strategy_spec_fingerprint(saved_strategy.spec) != strategy_spec_fingerprint(strategy_spec):
        return {
            "status": "BLOCKED",
            "reason": "SAVED_STRATEGY_SPEC_MISMATCH",
            "strategy_id": str(strategy_id),
            "research_only": True,
            "live_trading_allowed": False,
        }
    strategy_spec = saved_strategy.spec
elif saved_strategy is not None:
    strategy_spec = saved_strategy.spec
```

Also reject:

```python
if strategy_spec is not None and strategy_id and str(strategy_id) != strategy_spec.strategy_id:
    return {
        "status": "BLOCKED",
        "reason": "CONFIG_SPEC_MISMATCH",
        "unsupported_fields": ["config.strategy_id"],
        "research_only": True,
        "live_trading_allowed": False,
    }
```

This logic must run before capability checks and universe resolution.

- [ ] **Step 6: Add the Agent boundary regression**

Append to `tests/unit/agent/test_saved_generated_strategy_backtest_guard.py`:

```python
def test_inline_spec_cannot_replace_saved_strategy(tmp_path, monkeypatch) -> None:
    # Save strategy_id="saved_value" with top_n=10 using the module's existing helpers.
    # Submit inline strategy_spec with strategy_id="saved_value" and top_n=20.
    # Monkeypatch universe resolution to raise if reached.

    result = strategy_tools._run_backtest(
        {
            "strategy_id": "saved_value",
            "strategy_spec": conflicting_spec.model_dump(mode="json"),
            "start_date": "20240101",
            "end_date": "20240331",
        },
        ToolContext(run_id="saved-spec-conflict"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "SAVED_STRATEGY_SPEC_MISMATCH"
```

Use full executable code matching the fixture helpers already in that file; do not leave the setup as comments in the committed test.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/unit/strategy/test_saved_strategy_identity.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/strategy/test_backtest_config_spec_consistency.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/strategy/execution_adapter.py \
        src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/strategy/test_saved_strategy_identity.py \
        tests/unit/agent/test_saved_generated_strategy_backtest_guard.py
git commit -m "fix(strategy): lock saved strategy identity"
```

---

# Task 4: Require Evidence for Trade-State Fields

**Files:**
- Modify: `src/qmt_agent_trader/data/bars.py`
- Modify: `src/qmt_agent_trader/factors/input_panel.py`
- Create: `tests/unit/data/test_trade_state_evidence.py`
- Modify: `tests/unit/strategy/test_backtest_factor_input_panel.py`

**Interfaces:**
- Produces:
  - `TRADE_STATE_SOURCE_NOT_READY`
  - `TRADE_STATE_PARTIAL_COVERAGE`
  - bar attributes `trade_state_quality`
- Required sources:
  - `tushare/suspend_d` for `suspended`;
  - `tushare/stk_limit` for `limit_up` and `limit_down`;
  - `tushare/namechange` for historical `st`.
- Missing required source datasets do not become all-`False` state columns.
- `stk_limit` must have one row for each stock bar date that may be executed.
- Sparse `suspend_d` and interval `namechange` datasets may legitimately have no matching row after the source dataset itself is present.

- [ ] **Step 1: Add missing-source tests**

Create `tests/unit/data/test_trade_state_evidence.py`:

```python
import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake


def _lake_with_daily(tmp_path) -> DataLake:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1_000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )
    return lake


def test_missing_trade_state_sources_fail_closed(tmp_path) -> None:
    lake = _lake_with_daily(tmp_path)

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_daily_bars(
            lake,
            start="20240102",
            end="20240102",
            symbols=["000001.SZ"],
        )

    assert exc_info.value.code == "TRADE_STATE_SOURCE_NOT_READY"
    assert set(exc_info.value.details["missing_datasets"]) == {
        "tushare/suspend_d",
        "tushare/stk_limit",
        "tushare/namechange",
    }


def test_missing_stk_limit_row_fails_closed(tmp_path) -> None:
    lake = _lake_with_daily(tmp_path)
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
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240102",
                    "up_limit": 20.0,
                    "down_limit": 18.0,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_daily_bars(
            lake,
            start="20240102",
            end="20240102",
            symbols=["000001.SZ"],
        )

    assert exc_info.value.code == "TRADE_STATE_PARTIAL_COVERAGE"
    assert exc_info.value.details["field"] == "limit_up_down"
```

- [ ] **Step 2: Run the trade-state tests**

```bash
uv run pytest tests/unit/data/test_trade_state_evidence.py -q
```

Expected: FAIL because missing sources currently become `False`.

- [ ] **Step 3: Stop creating unproven state values in normalization**

In `normalize_tushare_daily()` replace:

```python
for column in ["suspended", "limit_up", "limit_down", "st"]:
    if column not in data.columns:
        data[column] = False
```

with:

```python
for column in ["suspended", "limit_up", "limit_down", "st"]:
    if column not in data.columns:
        data[column] = pd.NA
```

Do not cast these columns to `bool` until their evidence has been resolved.

- [ ] **Step 4: Validate source dataset availability**

Add:

```python
_REQUIRED_TRADE_STATE_DATASETS = {
    "suspended": "tushare/suspend_d",
    "limit_up_down": "tushare/stk_limit",
    "st": "tushare/namechange",
}


def _require_trade_state_sources(lake: DataLake) -> None:
    missing = [
        dataset
        for dataset in _REQUIRED_TRADE_STATE_DATASETS.values()
        if not lake.dataset_path("raw", dataset).exists()
    ]
    if missing:
        raise BacktestDataIntegrityError(
            code="TRADE_STATE_SOURCE_NOT_READY",
            message="required trade-state source datasets are unavailable",
            field="trade_state",
            details={"missing_datasets": missing},
        )
```

Call it in `load_daily_bars()` whenever `include_trade_state=True`, before reading/enriching state datasets.

- [ ] **Step 5: Require exact `stk_limit` coverage**

After normalizing the `stk_limit` source, validate source uniqueness and compare keys:

```python
bar_keys = set(
    zip(
        bars["symbol"].astype(str),
        bars["trade_date"],
        strict=False,
    )
)
limit_keys = set(
    zip(
        limits["symbol"].astype(str),
        limits["trade_date"],
        strict=False,
    )
)
missing_limit_keys = sorted(bar_keys - limit_keys)
if missing_limit_keys:
    raise BacktestDataIntegrityError(
        code="TRADE_STATE_PARTIAL_COVERAGE",
        message="stk_limit does not cover every executable symbol-date bar",
        field="trade_state",
        symbols=tuple(sorted({symbol for symbol, _ in missing_limit_keys})),
        details={
            "field": "limit_up_down",
            "missing_key_count": len(missing_limit_keys),
            "sample": [
                {"symbol": symbol, "trade_date": day.isoformat()}
                for symbol, day in missing_limit_keys[:20]
            ],
        },
    )
```

Use `require_unique_symbol_dates()` on `stk_limit` before building `limit_keys`.

- [ ] **Step 6: Resolve sparse and interval states explicitly**

After source validation:

- initialize `suspended=False` because the sparse `suspend_d` dataset is present and absence of a matching suspension event means not suspended;
- initialize `st=False` because the historical `namechange` interval dataset is present and absence of an overlapping ST interval means not ST;
- derive `limit_up`/`limit_down` only from exact `stk_limit` rows;
- cast all four columns to non-null booleans only after these operations.

Record:

```python
result.attrs["trade_state_quality"] = {
    "suspended": {"source": "raw/tushare/suspend_d", "complete": True},
    "limit_up": {"source": "raw/tushare/stk_limit", "complete": True},
    "limit_down": {"source": "raw/tushare/stk_limit", "complete": True},
    "st": {"source": "raw/tushare/namechange", "complete": True},
}
```

- [ ] **Step 7: Surface trade-state evidence in panel metadata**

In `build_target_frequency_panel()` after loading bars:

```python
metadata["trade_state_quality"] = dict(
    panel.attrs.get("trade_state_quality") or {}
)
```

If any required field lacks `complete=True`, raise `BacktestDataIntegrityError` rather than returning a completed panel.

- [ ] **Step 8: Add the valid-source test**

Append to `tests/unit/data/test_trade_state_evidence.py`:

```python
def test_complete_trade_state_sources_produce_boolean_evidence(tmp_path) -> None:
    lake = _lake_with_daily(tmp_path)
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
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )

    bars = load_daily_bars(
        lake,
        start="20240102",
        end="20240102",
        symbols=["000001.SZ"],
    )

    assert bars[["suspended", "limit_up", "limit_down", "st"]].isna().sum().sum() == 0
    assert bars.attrs["trade_state_quality"]["limit_up"]["complete"] is True
```

- [ ] **Step 9: Run focused tests**

```bash
uv run pytest \
  tests/unit/data/test_trade_state_evidence.py \
  tests/unit/factors/test_input_panel.py \
  tests/unit/strategy/test_backtest_factor_input_panel.py \
  tests/unit/backtest/test_research_runner_valuation.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/qmt_agent_trader/data/bars.py \
        src/qmt_agent_trader/factors/input_panel.py \
        tests/unit/data/test_trade_state_evidence.py \
        tests/unit/strategy/test_backtest_factor_input_panel.py
git commit -m "fix(data): require trade state evidence"
```

---

# Task 5: Load Factor Warm-Up History Before the Requested Start

**Files:**
- Modify: `src/qmt_agent_trader/data/trading_calendar.py`
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/strategy/test_backtest_warmup.py`
- Modify: `tests/unit/backtest/test_expected_sessions.py`
- Modify: `tests/unit/backtest/test_research_runner_timeline.py`

**Interfaces:**
- Produces:
  - `load_session_window(lake, *, start, end, warmup_sessions, exchanges) -> TradingSessionWindow`
  - `TradingSessionWindow.warmup_dates`
  - `TradingSessionWindow.expected_dates`
  - `TradingSessionWindow.panel_start`
- Runner may receive bars before the first expected trade date only.
- Warm-up bars feed factor calculation but never enter execution schedule, equity points, or performance metrics.

- [ ] **Step 1: Add the calendar-window model and tests**

In `tests/unit/strategy/test_backtest_warmup.py`:

```python
from datetime import date

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.trading_calendar import load_session_window


def test_session_window_resolves_prior_open_days(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240101", "is_open": 0},
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240104", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    window = load_session_window(
        lake,
        start="20240104",
        end="20240105",
        warmup_sessions=2,
        exchanges=("SSE",),
    )

    assert window.warmup_dates == (date(2024, 1, 2), date(2024, 1, 3))
    assert window.expected_dates == (date(2024, 1, 4), date(2024, 1, 5))
    assert window.panel_start == date(2024, 1, 2)
```

- [ ] **Step 2: Add insufficient-history behavior**

```python
def test_insufficient_warmup_history_fails_closed(tmp_path) -> None:
    # Calendar contains only one open session before start, request two.
    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_session_window(
            lake,
            start="20240104",
            end="20240105",
            warmup_sessions=2,
            exchanges=("SSE",),
        )

    assert exc_info.value.code == "INSUFFICIENT_FACTOR_WARMUP_HISTORY"
    assert exc_info.value.details["required_sessions"] == 2
    assert exc_info.value.details["available_sessions"] == 1
```

- [ ] **Step 3: Run the calendar-window tests**

```bash
uv run pytest tests/unit/strategy/test_backtest_warmup.py -q
```

Expected: FAIL because `load_session_window` does not exist.

- [ ] **Step 4: Implement the session window**

In `trading_calendar.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TradingSessionWindow:
    warmup_dates: tuple[date, ...]
    expected_dates: tuple[date, ...]

    @property
    def panel_start(self) -> date:
        if self.warmup_dates:
            return self.warmup_dates[0]
        return self.expected_dates[0]
```

Refactor calendar normalization into a private helper returning one daily state per date. Then implement:

```python
def load_session_window(
    lake: DataLake,
    *,
    start: str,
    end: str,
    warmup_sessions: int,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> TradingSessionWindow:
    if warmup_sessions < 0:
        raise ValueError("warmup_sessions must be non-negative")
    start_date = _parse_boundary(start)
    end_date = _parse_boundary(end)
    states = _load_normalized_calendar_states(
        lake,
        exchanges=exchanges,
        coverage_start=None,
        coverage_end=end_date,
    )
    expected_dates = tuple(
        day
        for day, is_open in states.items()
        if start_date <= day <= end_date and is_open == 1
    )
    prior_open = [
        day
        for day, is_open in states.items()
        if day < start_date and is_open == 1
    ]
    warmup_dates = tuple(prior_open[-warmup_sessions:]) if warmup_sessions else ()
    if len(warmup_dates) != warmup_sessions:
        raise BacktestDataIntegrityError(
            code="INSUFFICIENT_FACTOR_WARMUP_HISTORY",
            message="trade calendar lacks enough prior open sessions for factor warm-up",
            field="trade_cal",
            details={
                "required_sessions": warmup_sessions,
                "available_sessions": len(prior_open),
                "start": start_date.isoformat(),
            },
        )
    if not expected_dates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_EMPTY",
            message="requested range contains no open sessions",
            field="trade_cal",
            details={"start": start, "end": end},
        )
    return TradingSessionWindow(
        warmup_dates=warmup_dates,
        expected_dates=expected_dates,
    )
```

Keep `load_open_sessions()` as:

```python
return load_session_window(
    lake,
    start=start,
    end=end,
    warmup_sessions=0,
    exchanges=exchanges,
).expected_dates
```

Preserve the existing natural-date coverage validation for the requested interval.

- [ ] **Step 5: Resolve maximum factor lookback in the adapter**

Add:

```python
def _maximum_factor_lookback(
    registry: FactorRegistry | None,
    factor_ids: list[str],
) -> int:
    if registry is None:
        return 0
    lookbacks = []
    for factor_id in factor_ids:
        saved = registry.get_factor(factor_id)
        if saved is None:
            continue
        lookbacks.append(max(0, int(saved.lookback)))
    return max(lookbacks, default=0)
```

After resolving the execution factor registry and before building the panel:

```python
warmup_sessions = _maximum_factor_lookback(
    factor_registry,
    requested_factor_ids,
)
session_window = load_session_window(
    lake,
    start=config.start_date,
    end=config.end_date,
    warmup_sessions=warmup_sessions,
)
panel_start = f"{session_window.panel_start:%Y%m%d}"
```

Build the panel with:

```python
target_start=panel_start,
target_end=config.end_date,
```

Pass only:

```python
expected_trade_dates=session_window.expected_dates,
```

to `FactorRankResearchConfig`.

- [ ] **Step 6: Allow only pre-start warm-up bars in the runner**

In `FactorRankResearchRunner.__init__()` replace the strict unexpected-date rejection with:

```python
first_expected = config.expected_trade_dates[0]
last_expected = config.expected_trade_dates[-1]
unexpected_dates = sorted(
    day
    for day in observed_dates
    if day > last_expected
)
interior_non_session_dates = sorted(
    day
    for day in observed_dates
    if first_expected <= day <= last_expected and day not in expected_dates
)
if unexpected_dates or interior_non_session_dates:
    raise BacktestDataIntegrityError(
        code="UNEXPECTED_MARKET_SESSION",
        message="market bars contain non-calendar dates inside or after the backtest window",
        field="trade_date",
        details={
            "unexpected_dates": [
                f"{item:%Y-%m-%d}"
                for item in [*interior_non_session_dates, *unexpected_dates]
            ]
        },
    )
```

Compute factors on all bars, but create execution bar indexes from expected dates only:

```python
execution_bars = self.bars[
    self.bars["trade_date"].isin(config.expected_trade_dates)
].copy()
self._bars_by_date_symbol = {
    trade_date: frame.set_index("symbol", drop=False)
    for trade_date, frame in execution_bars.groupby("trade_date", sort=True)
}
```

- [ ] **Step 7: Add the warm-up factor regression**

In `tests/unit/strategy/test_backtest_warmup.py`, monkeypatch the panel builder and runner:

```python
def test_adapter_loads_factor_lookback_before_requested_start(
    tmp_path,
    monkeypatch,
) -> None:
    observed = {}

    def fake_panel(_lake, **kwargs):
        observed["target_start"] = kwargs["target_start"]
        rows = []
        for day in ("20240102", "20240103", "20240104", "20240105"):
            rows.append(
                {
                    "symbol": "000001.SZ",
                    "trade_date": pd.Timestamp(day).date(),
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 100.0,
                    "amount": 1_000.0,
                    "turnover": 0.01,
                    "suspended": False,
                    "limit_up": False,
                    "limit_down": False,
                    "st": False,
                }
            )
        return pd.DataFrame(rows), complete_panel_metadata()

    class FakeRunner:
        def __init__(self, bars, config):
            observed["runner_dates"] = config.expected_trade_dates
            observed["bar_dates"] = tuple(sorted(bars["trade_date"].unique()))
            self.factor_frame = pd.DataFrame()
            self.bars = bars

        def run(self, scenario):
            raise BacktestDataIntegrityError(
                code="fixture_stop",
                message="stop after warm-up assertions",
            )

    # Register a saved factor with lookback=2 or monkeypatch registry.get_factor().
    # Provide trade_cal covering 20240101 through 20240105.
    monkeypatch.setattr(execution_adapter, "build_target_frequency_panel", fake_panel)
    monkeypatch.setattr(execution_adapter, "FactorRankResearchRunner", FakeRunner)

    with pytest.raises(BacktestDataIntegrityError, match="stop after warm-up"):
        run_strategy_backtest(...)

    assert observed["target_start"] == "20240102"
    assert observed["runner_dates"] == (
        date(2024, 1, 4),
        date(2024, 1, 5),
    )
    assert observed["bar_dates"] == (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    )
```

Replace the ellipsis and setup comments with the module's actual fixtures and constructors in the committed test.

- [ ] **Step 8: Add runner warm-up timeline tests**

In `tests/unit/backtest/test_research_runner_timeline.py`:

```python
def test_pre_start_bars_are_factor_inputs_not_equity_dates(monkeypatch) -> None:
    bars = bars_for_dates(
        [
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
        ]
    )
    expected = (date(2024, 1, 4), date(2024, 1, 5))
    factors = factor_frame_for_dates(bars["trade_date"].unique())
    monkeypatch.setattr(research_runner, "compute_factor_frame", lambda *_args, **_kwargs: factors)

    result = FactorRankResearchRunner(
        bars,
        FactorRankResearchConfig(
            factor_name="fixture",
            expected_trade_dates=expected,
            top_n=1,
            max_single_position_pct=1.0,
        ),
    ).run(SensitivityScenario(execution_delay_days=1, top_n=1))

    assert [point.trade_date for point in result.equity_points] == [
        "2024-01-04",
        "2024-01-05",
    ]
```

Use the file's actual fixture helpers or add complete helpers in the test file.

- [ ] **Step 9: Expose warm-up metadata**

Add to `data_window`:

```python
"factor_warmup_sessions": warmup_sessions,
"factor_warmup_start": panel_start,
"requested_start": config.start_date,
```

Keep `actual_start` and `actual_end` for the loaded panel, and add:

```python
"performance_start": f"{session_window.expected_dates[0]:%Y%m%d}",
"performance_end": f"{session_window.expected_dates[-1]:%Y%m%d}",
```

- [ ] **Step 10: Run focused tests**

```bash
uv run pytest \
  tests/unit/strategy/test_backtest_warmup.py \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/backtest/test_expected_sessions.py \
  tests/unit/backtest/test_research_runner_timeline.py \
  tests/unit/strategy/test_backtest_factor_input_panel.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/qmt_agent_trader/data/trading_calendar.py \
        src/qmt_agent_trader/strategy/execution_adapter.py \
        src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/strategy/test_backtest_warmup.py \
        tests/unit/backtest/test_expected_sessions.py \
        tests/unit/backtest/test_research_runner_timeline.py
git commit -m "fix(backtest): preload factor warmup history"
```

---

# Task 6: Put Signal-Availability Counts in Canonical Data Quality

**Files:**
- Modify: `src/qmt_agent_trader/backtest/research_models.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Modify: `tests/unit/backtest/test_research_models.py`
- Modify: `tests/unit/backtest/test_research_runner_signal_availability.py`
- Modify: `tests/unit/strategy/test_backtest_report_schema.py`

**Interfaces:**
- `ResearchDataQuality` owns:
  - `scheduled_rebalance_count`
  - `available_signal_count`
  - `signal_unavailable_count`
- Legacy top-level serialization remains for one migration cycle but is sourced from `data_quality`.
- `StrategyBacktestResult.data_quality` and schema 2.0 reports expose the counts.

- [ ] **Step 1: Add the canonical serialization regression**

Append to `tests/unit/backtest/test_research_models.py`:

```python
def test_signal_availability_counts_are_canonical_data_quality() -> None:
    result = FactorRankResearchResult(
        metrics=SensitivityMetrics(total_return=0.0),
        trades=(),
        equity_points=(),
        rebalance_points=(),
        data_quality=ResearchDataQuality(
            scheduled_rebalance_count=5,
            available_signal_count=3,
            signal_unavailable_count=2,
        ),
    )

    payload = result.as_dict()

    assert payload["data_quality"]["scheduled_rebalance_count"] == 5
    assert payload["data_quality"]["available_signal_count"] == 3
    assert payload["data_quality"]["signal_unavailable_count"] == 2
    assert payload["scheduled_rebalance_count"] == 5
```

- [ ] **Step 2: Run the model test**

```bash
uv run pytest tests/unit/backtest/test_research_models.py -q
```

Expected: FAIL because the counts currently live on `FactorRankResearchResult`.

- [ ] **Step 3: Move count fields into `ResearchDataQuality`**

```python
@dataclass(frozen=True)
class ResearchDataQuality:
    validated_valuation_dates: int = 0
    low_cross_section_dates: tuple[str, ...] = ()
    rejected_order_count: int = 0
    warnings: tuple[str, ...] = ()
    scheduled_rebalance_count: int = 0
    available_signal_count: int = 0
    signal_unavailable_count: int = 0
```

Remove the three stored fields from `FactorRankResearchResult`, then add compatibility properties:

```python
@property
def scheduled_rebalance_count(self) -> int:
    return self.data_quality.scheduled_rebalance_count

@property
def available_signal_count(self) -> int:
    return self.data_quality.available_signal_count

@property
def signal_unavailable_count(self) -> int:
    return self.data_quality.signal_unavailable_count
```

In `as_dict()`, keep the top-level migration keys, sourced from these properties.

- [ ] **Step 4: Populate counts in the runner**

Construct:

```python
data_quality=ResearchDataQuality(
    validated_valuation_dates=len(equity_points),
    rejected_order_count=rejected_orders,
    scheduled_rebalance_count=len(execution_schedule),
    available_signal_count=len(available_signals),
    signal_unavailable_count=len(unavailable_signals),
),
```

Remove the three result-constructor keyword arguments.

- [ ] **Step 5: Assert schema 2.0 canonical evidence**

Append to `tests/unit/strategy/test_backtest_report_schema.py`:

```python
def test_canonical_data_quality_exposes_signal_counts() -> None:
    result = {
        "data_quality": {
            "validated_valuation_dates": 4,
            "scheduled_rebalance_count": 3,
            "available_signal_count": 2,
            "signal_unavailable_count": 1,
        },
        "equity_points": [],
        "rebalance_points": [],
        "trades": [],
    }
    evidence = _canonical_result_evidence(result, {})

    assert evidence["data_quality"]["scheduled_rebalance_count"] == 3
    assert evidence["data_quality"]["available_signal_count"] == 2
    assert evidence["data_quality"]["signal_unavailable_count"] == 1
```

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/unit/backtest/test_research_models.py \
  tests/unit/backtest/test_research_runner_signal_availability.py \
  tests/unit/strategy/test_backtest_report_schema.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/backtest/research_models.py \
        src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/backtest/test_research_models.py \
        tests/unit/backtest/test_research_runner_signal_availability.py \
        tests/unit/strategy/test_backtest_report_schema.py
git commit -m "fix(backtest): expose canonical signal quality counts"
```

---

# Task 7: Make Universe Ranking Deterministic at Ties

**Files:**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Ranking uses `[ranking.field, "symbol"]`.
- Ranking direction applies only to the primary field.
- Symbol tie-break is always ascending.
- Stable sorting is explicit.

- [ ] **Step 1: Add the tie-break regression**

Append to `tests/unit/universe/test_resolver.py`:

```python
def test_ranked_universe_ties_use_symbol_ascending_tiebreak() -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked",
            "name": "Ranked",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {"field": "avg_amount_20d", "ascending": False},
            "max_symbols": 2,
        }
    )
    frame = pd.DataFrame(
        {
            "symbol": ["000003.SZ", "000001.SZ", "000002.SZ"],
            "avg_amount_20d": [100.0, 100.0, 100.0],
        }
    )

    ranked = UniverseResolver._apply_ranking(
        object(),
        frame,
        spec,
    )
    symbols = _ordered_unique_symbols(ranked, spec)
    selected, _ = _apply_limit(symbols, spec=spec, limit=None)

    assert selected == ["000001.SZ", "000002.SZ"]
```

Prefer constructing a real resolver if the method invocation style above conflicts with linting in the current module.

- [ ] **Step 2: Run the resolver test**

```bash
uv run pytest tests/unit/universe/test_resolver.py -q
```

Expected: FAIL because tie ordering currently inherits input order.

- [ ] **Step 3: Implement stable tie-breaking**

Replace `_apply_ranking()` sort with:

```python
ranked = frame.sort_values(
    [ranking.field, "symbol"],
    ascending=[ranking.ascending, True],
    na_position="last",
    kind="stable",
)
```

Before sorting, if `"symbol"` is absent, return an empty frame or raise the existing universe-integrity error used for malformed candidate data. Do not synthesize a symbol.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_resolver.py \
  tests/unit/strategy/test_backtest_rolling_universe.py \
  tests/unit/strategy/test_backtest_snapshot_universe.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_resolver.py
git commit -m "fix(universe): stabilize ranking ties"
```

---

# Task 8: Update Documentation and Run the Final Local Gate

**Files:**
- Modify: `docs/backtest/factor-rank-adapter.md`

**Interfaces:**
- Documentation must describe the implemented behavior exactly.
- No CI configuration is part of this task.

- [ ] **Step 1: Update the adapter documentation**

Add the following contracts:

1. Raw market and exact factor-source duplicates are rejected before normalization or joins.
2. `rebalance_frequency` controls the strategy and must match an existing spec.
3. `universe_rebalance_frequency` is the only independent rolling-universe cadence input.
4. Saved Registry strategy identity and spec fingerprint cannot be replaced by an inline spec.
5. Missing `suspend_d`, `stk_limit`, or historical ST source evidence blocks execution.
6. Exact `stk_limit` coverage is required for executable stock bars.
7. Factor panels load the maximum declared factor lookback before the requested start.
8. Warm-up dates are excluded from the ledger, equity curve, and metrics.
9. Signal-availability counts live in canonical `data_quality`.
10. Universe ranking ties use ascending symbol order.

Delete or rewrite any statement claiming all trade-state columns are safe merely because they exist in the normalized frame.

- [ ] **Step 2: Run all new focused tests**

```bash
uv run pytest \
  tests/unit/data/test_data_integrity.py \
  tests/unit/data/test_trade_state_evidence.py \
  tests/unit/factors/test_input_panel.py \
  tests/unit/agent/test_agent_backtest_config_spec_consistency.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/strategy/test_saved_strategy_identity.py \
  tests/unit/strategy/test_backtest_warmup.py \
  tests/unit/backtest/test_duplicate_inputs.py \
  tests/unit/backtest/test_research_runner_signal_availability.py \
  tests/unit/backtest/test_research_runner_timeline.py \
  tests/unit/backtest/test_research_models.py \
  tests/unit/strategy/test_backtest_report_schema.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 3: Run affected subsystem suites**

```bash
uv run pytest \
  tests/unit/backtest \
  tests/unit/data \
  tests/unit/factors \
  tests/unit/strategy \
  tests/unit/universe \
  tests/unit/agent/test_agent_backtest_config_spec_consistency.py \
  tests/unit/agent/test_backtest_integrity_error_boundary.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py -q
```

Expected: PASS.

- [ ] **Step 4: Verify silent deduplication is gone from governed paths**

```bash
rg -n "drop_duplicates" \
  src/qmt_agent_trader/data/bars.py \
  src/qmt_agent_trader/factors/input_panel.py \
  src/qmt_agent_trader/backtest/research_runner.py
```

Review every result:

- no market or exact-source symbol-date identity may be resolved with `keep="last"` or `keep="first"`;
- deduplication used only after an explicit uniqueness validator, or for non-identity presentation data, must be documented in code.

- [ ] **Step 5: Verify broad exception swallowing is absent**

```bash
rg -n "except Exception" \
  src/qmt_agent_trader/backtest \
  src/qmt_agent_trader/data/bars.py \
  src/qmt_agent_trader/factors/input_panel.py \
  src/qmt_agent_trader/strategy/execution_adapter.py \
  src/qmt_agent_trader/agent/tools/strategy_tools.py
```

No broad catch may normalize adapter, runner, calendar, data-integrity, or Registry identity failures. Existing artifact-enumeration and code-generation boundaries may remain only when they are outside backtest execution and explicitly report the failure.

- [ ] **Step 6: Run repository quality gates**

```bash
make check
```

Expected: exit code `0`.

- [ ] **Step 7: Commit documentation**

```bash
git add docs/backtest/factor-rank-adapter.md
git commit -m "docs(backtest): document final fail closed contracts"
```

---

# Final Acceptance Checklist

## Raw data integrity

- [ ] Duplicate raw daily bars fail before normalization.
- [ ] Duplicate exact factor-input rows fail before joining.
- [ ] Identical duplicates are errors, not harmless deduplication.
- [ ] The runner retains its own uniqueness checks as defense in depth.

## Strategy and universe semantics

- [ ] `rebalance_frequency` affects strategy semantics only.
- [ ] A conflicting strategy frequency returns `CONFIG_SPEC_MISMATCH`.
- [ ] Factor-only temporary strategies persist the requested frequency in their spec.
- [ ] `universe_rebalance_frequency` independently controls rolling-universe snapshots.
- [ ] The default rolling-universe cadence is the authoritative strategy frequency.

## Saved strategy identity

- [ ] `config.strategy_id` equals `StrategySpec.strategy_id`.
- [ ] An inline spec cannot replace a saved Registry spec.
- [ ] Registry and inline spec conflicts return `SAVED_STRATEGY_SPEC_MISMATCH`.
- [ ] Report and artifact identity always match the executed Registry strategy.

## Trade-state evidence

- [ ] Missing required trade-state source datasets fail closed.
- [ ] Missing `stk_limit` symbol-date coverage fails closed.
- [ ] No unknown trade-state value is converted to `False`.
- [ ] Completed panels expose source and completeness evidence.

## Factor warm-up

- [ ] Maximum declared lookback is resolved across all requested factors.
- [ ] The panel starts far enough before the requested date.
- [ ] Insufficient prior sessions raise `INSUFFICIENT_FACTOR_WARMUP_HISTORY`.
- [ ] Warm-up bars participate in factor computation only.
- [ ] No warm-up date appears in trades, equity points, or metrics.
- [ ] Report metadata distinguishes panel start from performance start.

## Canonical evidence

- [ ] Scheduled, available, and unavailable signal counts are in `data_quality`.
- [ ] Schema 2.0 Agent results expose those counts without reading legacy `payload`.
- [ ] Legacy top-level count keys, if retained, are sourced from canonical data quality.

## Universe determinism

- [ ] Ranking order survives deduplication and limits.
- [ ] Equal ranking values use symbol-ascending tie-breaking.
- [ ] Sorting is explicitly stable.

## Safety and verification

- [ ] Integrity failures create no completed report or successful cache entry.
- [ ] Unexpected software errors propagate.
- [ ] `research_only=True`.
- [ ] `live_trading_allowed=False`.
- [ ] All focused tests pass.
- [ ] Affected subsystem tests pass.
- [ ] `make check` passes.
- [ ] Documentation matches implementation.

## Explicitly Out of Scope

- Historical extreme-drawdown replay.
- A new full DataLake-to-Agent end-to-end fixture.
- GitHub Actions creation or modification.
- Process-isolated execution of generated strategy Python.

## Expected Merge Decision

Keep the branch at `REQUEST_CHANGES` until every acceptance item and local command passes. After completion, perform one final static review of the branch before merging.
