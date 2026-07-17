# Factor-Rank PIT Rule Integrity Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining point-in-time universe gaps so liquidity rules cannot admit incomplete windows, ETF category selection cannot fall back to current or missing metadata, invalid effective intervals fail closed, profiling uses canonical datasets, ETF-only universes ignore unrelated stock metadata, and normalized index members are unique before selection.

**Architecture:** Keep all universe eligibility decisions behind explicit evidence gates. Reuse one shared liquidity-field contract for filters, rules, and ranking; require dated classification evidence for any historical category selection; validate interval order immediately after parsing; and normalize security identifiers before uniqueness checks. Keep the profiling path on the same canonical DataLake names and strategy contracts as production.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, pytest, existing `DataLake`, `UniverseResolver`, `StrategyBacktestConfig`, `UniverseSpec`, Ruff, mypy, and `uv`.

## Global Constraints

- Target branch: `codex/factor-rank-backtest-correctness`.
- Continue from reviewed head `5087e3c8f70603f608e4f269065947b5138a9311`.
- Save this plan as `docs/superpowers/plans/2026-07-17-factor-rank-pit-rule-integrity-repairs.md`.
- Create an isolated worktree before implementation.
- Follow TDD for every task: failing test, focused implementation, passing test, commit.
- Use one focused commit per task.
- Preserve `research_only=True` and `live_trading_allowed=False`.
- Unexpected programming errors must propagate.
- Data-integrity errors must remain typed and fail closed.
- Integrity failures must not write completed reports or successful cache entries.
- Do not add runtime dependencies.
- Do not add or modify GitHub Actions.
- Do not create a new full DataLake-to-Agent integration suite.
- Do not reproduce the historical extreme-drawdown run.
- Keep `BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v4"`.
- Bump only the engine semantic version because universe selection semantics change.
- Do not weaken the existing effective-session, trade-state, warm-up, Registry identity, ASOF ambiguity, or dataset-manifest contracts.

---

## File Responsibility Map

### Existing files to modify

- `src/qmt_agent_trader/universe/resolver.py`
  - Apply shared evidence eligibility to declarative rules.
  - Avoid loading stock master data for unrelated ETF-only universes.
  - Consume strict ETF classification support.

- `src/qmt_agent_trader/universe/models.py`
  - Require category values for `etf_category` selection.

- `src/qmt_agent_trader/universe/pit_metadata.py`
  - Validate listing and membership interval order.
  - Normalize index members before duplicate detection.

- `scripts/profile_research_tools.py`
  - Use canonical raw dataset names in discovery, bounds, and sampling.

- `src/qmt_agent_trader/agent/tools/strategy_tools.py`
  - Bump the engine semantic version.

- `docs/backtest/factor-rank-adapter.md`
  - Document strict rule eligibility, ETF category behavior, interval validation, and normalized index uniqueness.

### Existing tests to modify

- `tests/unit/universe/test_resolver.py`
- `tests/unit/universe/test_models.py`
- `tests/unit/universe/test_pit_security_master.py`
- `tests/unit/universe/test_index_membership_asof.py`
- `tests/unit/universe/test_exact_session_resolution.py`
- `tests/unit/scripts/test_profile_research_tools.py`
- `tests/unit/agent/test_backtest_cache_provenance.py`

---

### Task 1: Enforce Complete Liquidity Evidence for Declarative Rules

**Files:**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Produces `_field_evidence_eligible_rows(frame: pd.DataFrame, field: str) -> pd.DataFrame`.
- `_apply_rules()` uses the helper before applying every rule.
- `_apply_ranking()` reuses the same helper.
- For `avg_amount_20d`, eligibility requires a finite value and `amount_observation_count == 20`.
- For `avg_volume_20d`, eligibility requires a finite value and `volume_observation_count == 20`.

- [ ] **Step 1: Add the `ne` fail-open regression**

Append to `tests/unit/universe/test_resolver.py`:

```python
def test_liquidity_rule_ne_rejects_incomplete_window(tmp_path) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "amount-rule-ne",
            "name": "Amount rule ne",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "all",
                "rules": [
                    {
                        "field": "avg_amount_20d",
                        "operator": "ne",
                        "value": 0,
                    }
                ],
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_amount_20d": 1000.0,
                "amount_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_amount_20d": pd.NA,
                "amount_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, spec.selection.rules)

    assert observed["symbol"].tolist() == ["000001.SZ"]
```

- [ ] **Step 2: Add the `not_in` fail-open regression**

Append:

```python
def test_liquidity_rule_not_in_rejects_missing_evidence(tmp_path) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "volume-rule-not-in",
            "name": "Volume rule not in",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "all",
                "rules": [
                    {
                        "field": "avg_volume_20d",
                        "operator": "not_in",
                        "value": [0],
                    }
                ],
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_volume_20d": 100.0,
                "volume_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_volume_20d": pd.NA,
                "volume_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, spec.selection.rules)

    assert observed["symbol"].tolist() == ["000001.SZ"]
```

- [ ] **Step 3: Add the missing-count-column regression**

Append:

```python
def test_liquidity_rule_requires_observation_count_column(tmp_path) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "amount-rule-count",
            "name": "Amount rule count",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "all",
                "rules": [
                    {
                        "field": "avg_amount_20d",
                        "operator": "gt",
                        "value": 10,
                    }
                ],
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_amount_20d": 1000.0,
            }
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, spec.selection.rules)

    assert observed.empty
```

- [ ] **Step 4: Run the three regressions**

```bash
uv run pytest \
  tests/unit/universe/test_resolver.py::test_liquidity_rule_ne_rejects_incomplete_window \
  tests/unit/universe/test_resolver.py::test_liquidity_rule_not_in_rejects_missing_evidence \
  tests/unit/universe/test_resolver.py::test_liquidity_rule_requires_observation_count_column -q
```

Expected: FAIL because `_apply_rules()` currently applies `_rule_mask()` without evidence eligibility.

- [ ] **Step 5: Replace the ranking-only helper with a shared field helper**

In `src/qmt_agent_trader/universe/resolver.py`, replace `_ranking_eligible_rows()` with:

```python
def _field_evidence_eligible_rows(
    frame: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    if field not in frame.columns:
        return frame.iloc[0:0].copy()

    eligible = frame.copy()
    values = pd.to_numeric(
        eligible[field],
        errors="coerce",
    )
    eligible = eligible.loc[values.notna()].copy()

    count_field = {
        "avg_amount_20d": "amount_observation_count",
        "avg_volume_20d": "volume_observation_count",
    }.get(field)
    if count_field is None:
        return eligible
    if count_field not in eligible.columns:
        return eligible.iloc[0:0].copy()

    counts = pd.to_numeric(
        eligible[count_field],
        errors="coerce",
    )
    return eligible.loc[
        counts.eq(LIQUIDITY_WINDOW_SESSIONS)
    ].copy()
```

- [ ] **Step 6: Apply the helper before every rule**

Replace `_apply_rules()` with:

```python
def _apply_rules(
    self,
    frame: pd.DataFrame,
    rules: list[UniverseRule],
) -> pd.DataFrame:
    if frame.empty or not rules:
        return frame

    filtered = frame.copy()
    for rule in rules:
        filtered = _field_evidence_eligible_rows(
            filtered,
            rule.field,
        )
        if filtered.empty:
            return filtered
        series = filtered[rule.field]
        mask = _rule_mask(
            series,
            rule,
        ).fillna(False)
        filtered = filtered.loc[mask].copy()
    return filtered
```

- [ ] **Step 7: Reuse the helper for ranking**

In `_apply_ranking()`, replace:

```python
eligible = _ranking_eligible_rows(
    frame,
    ranking.field,
)
```

with:

```python
eligible = _field_evidence_eligible_rows(
    frame,
    ranking.field,
)
```

- [ ] **Step 8: Add a non-liquidity rule regression**

Append:

```python
def test_non_liquidity_rule_keeps_existing_semantics(tmp_path) -> None:
    rule = UniverseRule.model_validate(
        {
            "field": "market_cap",
            "operator": "gte",
            "value": 100,
        }
    )
    frame = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "market_cap": 100.0},
            {"symbol": "000002.SZ", "market_cap": 99.0},
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    observed = resolver._apply_rules(frame, [rule])

    assert observed["symbol"].tolist() == ["000001.SZ"]
```

- [ ] **Step 9: Run the resolver tests**

```bash
uv run pytest tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_resolver.py
git commit -m "fix(universe): gate liquidity rules on complete evidence"
```

---

### Task 2: Make ETF Category Selection Strict and Point-in-Time Safe

**Files:**
- Modify: `src/qmt_agent_trader/universe/models.py`
- Modify: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_models.py`
- Modify: `tests/unit/universe/test_exact_session_resolution.py`

**Interfaces:**
- `etf_category` requires non-empty category values in `theme_concepts` for backward-compatible schema use.
- `require_historical_classification_support()` treats `etf_category` as a dated classification mode.
- Without a dated classification frame, historical ETF category resolution raises `UNIVERSE_PIT_CLASSIFICATION_NOT_READY`.
- `_etf_category_candidates()` must never return all ETFs because category evidence is absent.

- [ ] **Step 1: Add the model validation regression**

Append to `tests/unit/universe/test_models.py`:

```python
def test_etf_category_requires_category_values() -> None:
    with pytest.raises(ValueError, match="etf_category selection requires categories"):
        UniverseSpec.model_validate(
            {
                "universe_id": "etf-category-empty",
                "name": "ETF category empty",
                "source": "user_defined",
                "asset_types": ["etf"],
                "selection": {
                    "mode": "etf_category",
                },
            }
        )
```

- [ ] **Step 2: Run the model regression**

```bash
uv run pytest \
  tests/unit/universe/test_models.py::test_etf_category_requires_category_values -q
```

Expected: FAIL because the validator does not currently require categories for `etf_category`.

- [ ] **Step 3: Extend the selection validator**

In `UniverseSelection._required_selection_values_present()`, add:

```python
if self.mode == "etf_category" and not self.theme_concepts:
    raise ValueError(
        "etf_category selection requires categories"
    )
```

Keep `theme_concepts` as the serialized field for compatibility; do not add a new schema field in this repair.

- [ ] **Step 4: Add the PIT classification regression**

Append to `tests/unit/universe/test_exact_session_resolution.py`:

```python
def test_etf_category_without_dated_classification_fails_closed(
    tmp_path,
) -> None:
    lake = DataLake(
        tmp_path / "lake",
        tmp_path / "research.duckdb",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": "20240102",
                    "is_open": 1,
                },
                {
                    "exchange": "SZSE",
                    "cal_date": "20240102",
                    "is_open": 1,
                },
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20240102",
                    "open": 3.5,
                    "high": 3.6,
                    "low": 3.4,
                    "close": 3.55,
                    "vol": 100.0,
                    "amount": 350.0,
                }
            ]
        ),
        "raw",
        "tushare/fund_daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20240102",
                    "up_limit": 3.85,
                    "down_limit": 3.15,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "etf-category",
            "name": "ETF category",
            "source": "user_defined",
            "asset_types": ["etf"],
            "selection": {
                "mode": "etf_category",
                "theme_concepts": ["broad_market"],
            },
            "filters": {"min_listed_days": 0},
        }
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake).build(
            spec,
            as_of_date="20240102",
        )

    assert exc_info.value.code == "UNIVERSE_PIT_CLASSIFICATION_NOT_READY"
    assert exc_info.value.field == "classification_history"
    assert exc_info.value.details["selection_mode"] == "etf_category"
```

- [ ] **Step 5: Run the PIT regression**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py::test_etf_category_without_dated_classification_fails_closed -q
```

Expected: FAIL because the PIT guard currently covers only `industry` and `theme`.

- [ ] **Step 6: Extend the PIT classification guard**

In `require_historical_classification_support()`, replace:

```python
if selection_mode not in {"industry", "theme"}:
    return
```

with:

```python
if selection_mode not in {
    "industry",
    "theme",
    "etf_category",
}:
    return
```

Keep the required dated columns:

```python
required = {
    "symbol",
    "effective_from",
    "effective_to",
}
```

- [ ] **Step 7: Remove the all-ETF fallback**

Replace `_etf_category_candidates()` with a defensive implementation that cannot be reached without dated evidence:

```python
def _etf_category_candidates(
    self,
    categories: list[str],
    recent: pd.DataFrame,
) -> pd.DataFrame:
    if not categories:
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_PIT_CLASSIFICATION_NOT_READY",
            message="ETF category selection lacks category values",
            field="classification_history",
            details={
                "selection_mode": "etf_category",
            },
        )
    raise BacktestUniverseIntegrityError(
        code="UNIVERSE_PIT_CLASSIFICATION_NOT_READY",
        message=(
            "historical ETF category selection requires dated "
            "classification evidence"
        ),
        field="classification_history",
        details={
            "selection_mode": "etf_category",
            "categories": list(categories),
        },
    )
```

This keeps the current project fail-closed until a real dated ETF classification source is implemented.

- [ ] **Step 8: Add a direct fallback regression**

Append:

```python
def test_etf_category_candidate_helper_never_falls_back_to_all_etfs(
    tmp_path,
) -> None:
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
    recent = pd.DataFrame(
        [
            {
                "symbol": "510300.SH",
                "asset_type": "etf",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        resolver._etf_category_candidates(
            ["broad_market"],
            recent,
        )

    assert exc_info.value.code == "UNIVERSE_PIT_CLASSIFICATION_NOT_READY"
```

- [ ] **Step 9: Run the focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_models.py \
  tests/unit/universe/test_exact_session_resolution.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/qmt_agent_trader/universe/models.py \
        src/qmt_agent_trader/universe/pit_metadata.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_models.py \
        tests/unit/universe/test_exact_session_resolution.py
git commit -m "fix(universe): block undated ETF category selection"
```

---

### Task 3: Reject Inverted Listing and Membership Intervals

**Files:**
- Modify: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `tests/unit/universe/test_pit_security_master.py`
- Modify: `tests/unit/universe/test_index_membership_asof.py`

**Interfaces:**
- Produces `_raise_invalid_interval(...) -> None`.
- `security_master_asof()` rejects `delist_date < list_date`.
- `index_interval_members_by_code_asof()` rejects `out_date <= in_date`.
- Error details include `invalid_row_count` and up to five identifying samples.

- [ ] **Step 1: Add the inverted listing-window regression**

Append to `tests/unit/universe/test_pit_security_master.py`:

```python
def test_security_master_rejects_inverted_listing_interval() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "Fixture",
                "list_date": "20250101",
                "delist_date": "20240101",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        security_master_asof(
            frame,
            date(2024, 6, 1),
        )

    assert exc_info.value.code == "UNIVERSE_SECURITY_MASTER_INVALID"
    assert exc_info.value.field == "raw/tushare/stock_basic"
    assert exc_info.value.details["invalid_row_count"] == 1
    assert exc_info.value.details["sample_keys"] == ["000001.SZ"]
```

- [ ] **Step 2: Add the inverted index-membership regression**

Append to `tests/unit/universe/test_index_membership_asof.py`:

```python
def test_index_member_rejects_inverted_effective_interval() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "20250101",
                "out_date": "20240101",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_interval_members_by_code_asof(
            frame,
            ["000300.SH"],
            date(2024, 6, 1),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_SOURCE_INVALID"
    assert exc_info.value.field == "raw/tushare/index_member"
    assert exc_info.value.details["invalid_row_count"] == 1
    assert exc_info.value.details["sample_keys"] == [
        "000300.SH:000001.SZ"
    ]
```

- [ ] **Step 3: Add the equal-boundary regression**

Append:

```python
def test_index_member_rejects_zero_length_interval() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "20240101",
                "out_date": "20240101",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_interval_members_by_code_asof(
            frame,
            ["000300.SH"],
            date(2024, 1, 1),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_SOURCE_INVALID"
```

- [ ] **Step 4: Run the regressions**

```bash
uv run pytest \
  tests/unit/universe/test_pit_security_master.py::test_security_master_rejects_inverted_listing_interval \
  tests/unit/universe/test_index_membership_asof.py::test_index_member_rejects_inverted_effective_interval \
  tests/unit/universe/test_index_membership_asof.py::test_index_member_rejects_zero_length_interval -q
```

Expected: FAIL because parsed dates are not checked for interval order.

- [ ] **Step 5: Add a shared interval-error helper**

In `pit_metadata.py`, add:

```python
def _raise_invalid_interval(
    *,
    invalid: pd.Series,
    code: str,
    message: str,
    field: str,
    sample_keys: pd.Series,
) -> None:
    if not invalid.any():
        return
    samples = (
        sample_keys.loc[invalid]
        .astype(str)
        .head(5)
        .tolist()
    )
    raise BacktestUniverseIntegrityError(
        code=code,
        message=message,
        field=field,
        details={
            "invalid_row_count": int(invalid.sum()),
            "sample_keys": samples,
        },
    )
```

- [ ] **Step 6: Validate stock listing intervals**

In `security_master_asof()`, immediately after parsing `delist_date`, add:

```python
invalid_interval = (
    data["delist_date"].notna()
    & data["list_date"].notna()
    & data["delist_date"].lt(data["list_date"])
)
_raise_invalid_interval(
    invalid=invalid_interval,
    code="UNIVERSE_SECURITY_MASTER_INVALID",
    message="stock_basic contains an inverted listing interval",
    field="raw/tushare/stock_basic",
    sample_keys=data["ts_code"],
)
```

Use `<`, not `<=`, because a same-day list and delist record is not automatically invalid under the current source contract.

- [ ] **Step 7: Validate index membership intervals**

In `index_interval_members_by_code_asof()`, immediately after parsing `out_date`, add:

```python
invalid_interval = (
    data["out_date"].notna()
    & data["in_date"].notna()
    & data["out_date"].le(data["in_date"])
)
member_keys = (
    data["index_code"].astype(str)
    + ":"
    + data["con_code"].astype(str)
)
_raise_invalid_interval(
    invalid=invalid_interval,
    code="INDEX_MEMBERSHIP_SOURCE_INVALID",
    message="index_member contains an invalid effective interval",
    field="raw/tushare/index_member",
    sample_keys=member_keys,
)
```

- [ ] **Step 8: Run PIT metadata tests**

```bash
uv run pytest \
  tests/unit/universe/test_pit_security_master.py \
  tests/unit/universe/test_index_membership_asof.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/qmt_agent_trader/universe/pit_metadata.py \
        tests/unit/universe/test_pit_security_master.py \
        tests/unit/universe/test_index_membership_asof.py
git commit -m "fix(universe): reject inverted PIT intervals"
```

---

### Task 4: Normalize Index Members Before Duplicate Validation

**Files:**
- Modify: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `tests/unit/universe/test_index_membership_asof.py`

**Interfaces:**
- Produces `_normalized_member_series(values: pd.Series, *, field: str) -> pd.Series`.
- Duplicate validation uses normalized member codes.
- Aliases such as `000001` and `000001.SZ` are duplicate evidence, not silently deduplicated.

- [ ] **Step 1: Add the weight-alias duplicate regression**

Append to `tests/unit/universe/test_index_membership_asof.py`:

```python
def test_index_weight_rejects_duplicate_members_after_normalization() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001",
                "trade_date": "20240201",
            },
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "trade_date": "20240201",
            },
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_weight_members_by_code_asof(
            frame,
            ["000300.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "DUPLICATE_UNIVERSE_SOURCE_KEY"
    assert exc_info.value.field == "raw/tushare/index_weight"
```

- [ ] **Step 2: Add the interval-alias duplicate regression**

Append:

```python
def test_index_interval_rejects_duplicate_members_after_normalization() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001",
                "in_date": "20200101",
                "out_date": None,
            },
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "20200101",
                "out_date": None,
            },
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_interval_members_by_code_asof(
            frame,
            ["000300.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "DUPLICATE_UNIVERSE_SOURCE_KEY"
    assert exc_info.value.field == "raw/tushare/index_member"
```

- [ ] **Step 3: Run the regressions**

```bash
uv run pytest \
  tests/unit/universe/test_index_membership_asof.py::test_index_weight_rejects_duplicate_members_after_normalization \
  tests/unit/universe/test_index_membership_asof.py::test_index_interval_rejects_duplicate_members_after_normalization -q
```

Expected: FAIL because uniqueness is checked before normalization.

- [ ] **Step 4: Replace list-only normalization with a series helper**

In `pit_metadata.py`, add:

```python
def _normalized_member_series(
    values: pd.Series,
    *,
    field: str,
) -> pd.Series:
    normalized = values.map(normalize_symbol)
    invalid = normalized.isna()
    if invalid.any():
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index membership contains invalid member identifiers",
            field=field,
            details={
                "invalid_row_count": int(invalid.sum()),
            },
        )
    return normalized.astype("string")
```

Replace `_normalized_member_values()` with:

```python
def _normalized_member_values(
    values: pd.Series,
    *,
    field: str,
) -> list[str]:
    return sorted(
        _normalized_member_series(
            values,
            field=field,
        )
        .astype(str)
        .unique()
        .tolist()
    )
```

- [ ] **Step 5: Normalize weight members before uniqueness checks**

After filtering `data` in `index_weight_members_by_code_asof()`, add:

```python
data["normalized_con_code"] = _normalized_member_series(
    data["con_code"],
    field="raw/tushare/index_weight.con_code",
)
```

Inside the latest-snapshot loop, change uniqueness validation to:

```python
require_unique_keys(
    snapshot,
    keys=(
        "index_code",
        "normalized_con_code",
        "trade_date",
    ),
    code="DUPLICATE_UNIVERSE_SOURCE_KEY",
    field="raw/tushare/index_weight",
)
members = sorted(
    snapshot["normalized_con_code"]
    .astype(str)
    .unique()
    .tolist()
)
```

- [ ] **Step 6: Normalize interval members before uniqueness checks**

Before building `active`, add:

```python
data["normalized_con_code"] = _normalized_member_series(
    data["con_code"],
    field="raw/tushare/index_member.con_code",
)
```

Change active uniqueness validation to:

```python
require_unique_keys(
    active,
    keys=(
        "index_code",
        "normalized_con_code",
    ),
    code="DUPLICATE_UNIVERSE_SOURCE_KEY",
    field="raw/tushare/index_member",
)
```

Build each member list from `normalized_con_code`:

```python
members = sorted(
    code_active["normalized_con_code"]
    .astype(str)
    .unique()
    .tolist()
)
```

- [ ] **Step 7: Run index membership tests**

```bash
uv run pytest tests/unit/universe/test_index_membership_asof.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/universe/pit_metadata.py \
        tests/unit/universe/test_index_membership_asof.py
git commit -m "fix(universe): validate normalized index member uniqueness"
```

---

### Task 5: Avoid Unrelated Stock-Master Validation for ETF-Only Universes

**Files:**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_exact_session_resolution.py`

**Interfaces:**
- Produces `_requires_stock_master(spec: UniverseSpec) -> bool`.
- ETF-only `all`, `explicit_symbols`, and future dated ETF classification paths do not read `stock_basic`.
- Any universe containing stock assets continues to use strict `security_master_asof()` validation.

- [ ] **Step 1: Add the ETF-only independence regression**

Append to `tests/unit/universe/test_exact_session_resolution.py`:

```python
def test_etf_only_snapshot_ignores_unrelated_invalid_stock_basic(
    tmp_path,
) -> None:
    lake = DataLake(
        tmp_path / "lake",
        tmp_path / "research.duckdb",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": "SSE",
                    "cal_date": "20240102",
                    "is_open": 1,
                },
                {
                    "exchange": "SZSE",
                    "cal_date": "20240102",
                    "is_open": 1,
                },
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20240102",
                    "open": 3.5,
                    "high": 3.6,
                    "low": 3.4,
                    "close": 3.55,
                    "vol": 100.0,
                    "amount": 350.0,
                }
            ]
        ),
        "raw",
        "tushare/fund_daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "20240102",
                    "up_limit": 3.85,
                    "down_limit": 3.15,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "list_date": "bad-date",
                }
            ]
        ),
        "raw",
        "tushare/stock_basic",
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "etf-only",
            "name": "ETF only",
            "source": "user_defined",
            "asset_types": ["etf"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )

    result = UniverseResolver(lake).build(
        spec,
        as_of_date="20240102",
    )

    assert result["status"] == "OK"
    assert result["symbols"] == ["510300.SH"]
```

- [ ] **Step 2: Run the regression**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py::test_etf_only_snapshot_ignores_unrelated_invalid_stock_basic -q
```

Expected: FAIL because `_resolve_for_date()` always calls `security_master_asof()`.

- [ ] **Step 3: Add the stock-master requirement helper**

In `resolver.py`, add:

```python
def _requires_stock_master(
    spec: UniverseSpec,
) -> bool:
    if "stock" in spec.asset_types:
        return True
    return spec.selection.mode in {
        "industry",
        "theme",
    }
```

- [ ] **Step 4: Gate stock-master loading**

Replace the unconditional call in `_resolve_for_date()` with:

```python
stock_basic = (
    security_master_asof(
        self._stock_basic(),
        session.effective_date,
    )
    if _requires_stock_master(spec)
    else pd.DataFrame(
        columns=[
            "symbol",
            "display_name",
            "list_date",
            "delist_date",
            "listed_as_of",
        ]
    )
)
```

- [ ] **Step 5: Add the mixed-universe strictness regression**

Append:

```python
def test_mixed_universe_still_validates_stock_basic(
    tmp_path,
    monkeypatch,
) -> None:
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
    monkeypatch.setattr(
        resolver,
        "_stock_basic",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "list_date": "bad-date",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        resolver_module,
        "_resolve_effective_session",
        lambda *_args, **_kwargs: resolver_module._ResolvedUniverseSession(
            requested_as_of=date(2024, 1, 2),
            effective_date=date(2024, 1, 2),
        ),
    )
    monkeypatch.setattr(
        resolver,
        "_load_recent_bars",
        lambda *_args: pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": date(2024, 1, 2),
                    "asset_type": "stock",
                },
                {
                    "symbol": "510300.SH",
                    "trade_date": date(2024, 1, 2),
                    "asset_type": "etf",
                },
            ]
        ),
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "mixed",
            "name": "Mixed",
            "source": "user_defined",
            "asset_types": ["stock", "etf"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        resolver._resolve_for_date(
            spec,
            as_of_date="20240102",
        )

    assert exc_info.value.code == "UNIVERSE_SECURITY_MASTER_INVALID"
```

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/unit/universe/test_exact_session_resolution.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_exact_session_resolution.py
git commit -m "fix(universe): skip stock master for ETF-only snapshots"
```

---

### Task 6: Use Canonical Dataset Names in the Profiling Script

**Files:**
- Modify: `scripts/profile_research_tools.py`
- Modify: `tests/unit/scripts/test_profile_research_tools.py`

**Interfaces:**
- Produces `MARKET_DATASETS: tuple[str, str]`.
- `_has_bars()`, `_date_bounds()`, and `_sample_symbols()` use canonical names.
- A normal DataLake with `tushare/daily` is detected as local data.

- [ ] **Step 1: Add the canonical-data discovery regression**

Append to `tests/unit/scripts/test_profile_research_tools.py`:

```python
def test_profile_has_bars_uses_canonical_dataset_names(tmp_path) -> None:
    module = _load_profile_module()
    lake = DataLake(
        tmp_path / "lake",
        tmp_path / "research.duckdb",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )

    assert module._has_bars(lake) is True
```

Add imports:

```python
import pandas as pd

from qmt_agent_trader.data.storage import DataLake
```

- [ ] **Step 2: Add bounds and sampling regressions**

Append:

```python
def test_profile_bounds_and_sampling_use_canonical_dataset_names(
    tmp_path,
) -> None:
    module = _load_profile_module()
    lake = DataLake(
        tmp_path / "lake",
        tmp_path / "research.duckdb",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240103",
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                },
            ]
        ),
        "raw",
        "tushare/daily",
    )

    assert module._date_bounds(lake) == {
        "start": "20240102",
        "end": "20240103",
    }
    assert module._sample_symbols(
        lake,
        start="20240101",
        end="20240131",
        limit=1,
    ) == ["000001.SZ"]
```

- [ ] **Step 3: Run the regressions**

```bash
uv run pytest tests/unit/scripts/test_profile_research_tools.py -q
```

Expected: FAIL because the script searches `tushare_daily` and `tushare_fund_daily`.

- [ ] **Step 4: Define one canonical dataset constant**

In `scripts/profile_research_tools.py`, add:

```python
MARKET_DATASETS: tuple[str, str] = (
    "tushare/daily",
    "tushare/fund_daily",
)
```

- [ ] **Step 5: Update all discovery loops**

Replace `_has_bars()` with:

```python
def _has_bars(lake: DataLake) -> bool:
    return any(
        lake.dataset_path(
            "raw",
            name,
        ).exists()
        for name in MARKET_DATASETS
    )
```

Replace both loops in `_date_bounds()` and `_sample_symbols()` with:

```python
for name in MARKET_DATASETS:
```

Update the `NO_LOCAL_DATA` message to:

```python
"raw tushare/daily or tushare/fund_daily parquet is not available"
```

- [ ] **Step 6: Run the script tests**

```bash
uv run pytest tests/unit/scripts/test_profile_research_tools.py -q
```

Expected: PASS.

- [ ] **Step 7: Run the profiling smoke command**

```bash
uv run python scripts/profile_research_tools.py --quick
```

Expected:

- exit code `0`;
- with no canonical market data, structured `NO_LOCAL_DATA` output;
- with canonical market data, the script proceeds beyond `_has_bars()` and does not fail due to old dataset names.

- [ ] **Step 8: Commit**

```bash
git add scripts/profile_research_tools.py \
        tests/unit/scripts/test_profile_research_tools.py
git commit -m "fix(tooling): use canonical profiling datasets"
```

---

### Task 7: Bump Cache Semantics, Update Documentation, and Verify

**Files:**
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Modify: `tests/unit/agent/test_backtest_cache_provenance.py`
- Modify: `docs/backtest/factor-rank-adapter.md`

**Interfaces:**
- Engine semantic version becomes `2026-07-universe-pit-rule-integrity-v5`.
- Cache schema remains `factor-rank-v4`.
- Documentation matches implemented fail-closed behavior.

- [ ] **Step 1: Update the semantic version test**

In `tests/unit/agent/test_backtest_cache_provenance.py`, change the expected value to:

```python
assert manifest["engine_semantic_version"] == (
    "2026-07-universe-pit-rule-integrity-v5"
)
```

- [ ] **Step 2: Run the semantic-version test**

```bash
uv run pytest \
  tests/unit/agent/test_backtest_cache_provenance.py -q
```

Expected: FAIL because production still reports the v4 semantic version.

- [ ] **Step 3: Bump the production semantic version**

In `src/qmt_agent_trader/agent/tools/strategy_tools.py`, set:

```python
BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v4"
BACKTEST_ENGINE_SEMANTIC_VERSION = (
    "2026-07-universe-pit-rule-integrity-v5"
)
```

- [ ] **Step 4: Document rule evidence eligibility**

Add to `docs/backtest/factor-rank-adapter.md` under liquidity completeness:

```markdown
The same evidence gate applies to declarative selection rules and ranking.
Rules using `avg_amount_20d` or `avg_volume_20d` cannot evaluate a row unless
its corresponding observation count is exactly 20 and the metric is non-null.
This includes negative operators such as `ne` and `not_in`.
```

- [ ] **Step 5: Document ETF category behavior**

Add under PIT classification:

```markdown
`etf_category` is a point-in-time classification mode. It requires explicit
category values and dated classification evidence. Current `fund_basic`
metadata, missing classification datasets, and empty category values never
fall back to all ETFs. Until a dated source is implemented, historical ETF
category requests fail with `UNIVERSE_PIT_CLASSIFICATION_NOT_READY`.
```

- [ ] **Step 6: Document interval and normalized-member validation**

Add:

```markdown
Listing intervals with `delist_date < list_date` and index membership intervals
with `out_date <= in_date` are invalid source evidence. Index members are
normalized before duplicate validation, so aliases such as `000001` and
`000001.SZ` are treated as conflicting duplicate records.
```

- [ ] **Step 7: Document ETF-only stock-master independence**

Add:

```markdown
Pure ETF universes do not read or validate unrelated stock master data. Mixed
or stock universes continue to require strict point-in-time `stock_basic`
evidence.
```

- [ ] **Step 8: Run the focused regression set**

```bash
uv run pytest \
  tests/unit/universe/test_models.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/universe/test_pit_security_master.py \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/scripts/test_profile_research_tools.py \
  tests/unit/agent/test_backtest_cache_provenance.py -q
```

Expected: PASS.

- [ ] **Step 9: Run the affected subsystem suites**

```bash
uv run pytest \
  tests/unit/data \
  tests/unit/universe \
  tests/unit/backtest \
  tests/unit/strategy \
  tests/unit/persistence/test_dataset_manifests.py \
  tests/unit/scripts/test_profile_research_tools.py \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_backtest_cache_provenance.py -q
```

Expected: PASS.

- [ ] **Step 10: Run the profiling smoke command**

```bash
uv run python scripts/profile_research_tools.py --quick
```

Expected: exit code `0`; canonical data is discovered when present.

- [ ] **Step 11: Verify old dataset names are gone**

```bash
rg -n "tushare_daily|tushare_fund_daily" \
  scripts/profile_research_tools.py \
  tests/unit/scripts/test_profile_research_tools.py
```

Expected: no matches.

- [ ] **Step 12: Verify rule eligibility is shared**

```bash
rg -n "_field_evidence_eligible_rows|_ranking_eligible_rows" \
  src/qmt_agent_trader/universe/resolver.py \
  tests/unit/universe
```

Expected:

- `_field_evidence_eligible_rows` is used by both `_apply_rules()` and `_apply_ranking()`;
- `_ranking_eligible_rows` has no remaining definition or call.

- [ ] **Step 13: Verify interval validation exists before filtering**

```bash
rg -n "invalid_interval|_raise_invalid_interval" \
  src/qmt_agent_trader/universe/pit_metadata.py
```

Expected: listing and index interval checks appear before as-of active filtering.

- [ ] **Step 14: Verify normalized uniqueness**

```bash
rg -n "normalized_con_code|DUPLICATE_UNIVERSE_SOURCE_KEY" \
  src/qmt_agent_trader/universe/pit_metadata.py \
  tests/unit/universe/test_index_membership_asof.py
```

Expected: both index sources validate keys using `normalized_con_code`.

- [ ] **Step 15: Verify semantic versions**

```bash
rg -n "BACKTEST_CACHE_SCHEMA_VERSION|BACKTEST_ENGINE_SEMANTIC_VERSION" \
  src/qmt_agent_trader/agent/tools/strategy_tools.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  docs/backtest/factor-rank-adapter.md
```

Expected:

```text
BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v4"
BACKTEST_ENGINE_SEMANTIC_VERSION = "2026-07-universe-pit-rule-integrity-v5"
```

- [ ] **Step 16: Run repository quality gates**

```bash
make check
```

Expected: exit code `0`.

- [ ] **Step 17: Commit**

```bash
git add src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/agent/test_backtest_cache_provenance.py \
        docs/backtest/factor-rank-adapter.md
git commit -m "docs(backtest): finalize PIT rule integrity contract"
```

---

## Final Acceptance Checklist

### Liquidity rules

- [ ] `avg_amount_20d` rules require a non-null value and exactly 20 amount observations.
- [ ] `avg_volume_20d` rules require a non-null value and exactly 20 volume observations.
- [ ] `ne` cannot admit a missing liquidity metric.
- [ ] `not_in` cannot admit a missing liquidity metric.
- [ ] Missing observation-count columns produce no eligible rows.
- [ ] Ranking and declarative rules use the same evidence helper.

### ETF categories

- [ ] `etf_category` requires explicit category values.
- [ ] Missing `fund_basic` never falls back to all ETFs.
- [ ] Current undated `fund_basic` metadata is not used for historical selection.
- [ ] Historical ETF category requests fail with `UNIVERSE_PIT_CLASSIFICATION_NOT_READY` until dated evidence exists.

### PIT intervals

- [ ] `delist_date < list_date` raises `UNIVERSE_SECURITY_MASTER_INVALID`.
- [ ] `out_date <= in_date` raises `INDEX_MEMBERSHIP_SOURCE_INVALID`.
- [ ] Error details include invalid counts and sample keys.
- [ ] Invalid intervals are rejected before as-of filtering.

### Index members

- [ ] Member codes are normalized before uniqueness checks.
- [ ] `000001` and `000001.SZ` conflict as duplicate evidence.
- [ ] Invalid normalized members fail closed.
- [ ] Valid members remain deterministically sorted.

### Asset isolation

- [ ] ETF-only universes ignore unrelated invalid `stock_basic` data.
- [ ] Mixed and stock universes still validate `stock_basic` strictly.
- [ ] Existing exact-session and mixed-asset coverage gates remain intact.

### Profiling

- [ ] Profiling uses `tushare/daily` and `tushare/fund_daily`.
- [ ] `_has_bars()`, `_date_bounds()`, and `_sample_symbols()` share one constant.
- [ ] The profiling smoke command reaches real probes when canonical data exists.
- [ ] Ad-hoc identity construction remains valid.

### Cache and safety

- [ ] Cache schema remains `factor-rank-v4`.
- [ ] Engine semantic version is `2026-07-universe-pit-rule-integrity-v5`.
- [ ] Old successful cache entries cannot survive the changed universe semantics.
- [ ] `research_only=True` remains unchanged.
- [ ] `live_trading_allowed=False` remains unchanged.
- [ ] Integrity failures create no completed report or successful cache entry.
- [ ] Unexpected software exceptions propagate.

### Verification

- [ ] Focused tests pass.
- [ ] Affected subsystem suites pass.
- [ ] Profiling smoke command exits successfully.
- [ ] `make check` passes.
- [ ] Documentation matches implementation.
- [ ] One final static branch review is performed against the new head.

## Explicitly Out of Scope

- Implementing a new dated ETF classification provider.
- Implementing historical industry or theme classification datasets.
- Creating a new full DataLake-to-Agent integration suite.
- Adding or changing GitHub Actions.
- Reproducing historical extreme drawdown results.
- Process-isolating generated strategy Python execution.
- Changing the backtest cache schema beyond `factor-rank-v4`.

## Expected Merge Decision

Keep `REQUEST_CHANGES` until every acceptance item and verification command
passes with recorded output. After implementation, run a fresh GitHub static
review against the new branch head before merging.
