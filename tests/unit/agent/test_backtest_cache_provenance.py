from __future__ import annotations

import pandas as pd

from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence import dataset_manifests
from qmt_agent_trader.persistence.provenance import fingerprint_path_tree
from qmt_agent_trader.strategy.execution_adapter import StrategyBacktestConfig
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategyKind,
    StrategySource,
    StrategySpec,
)


def _fixture_config() -> StrategyBacktestConfig:
    spec = StrategySpec(
        strategy_id="adhoc_factor_momentum_20d",
        name="Factor baseline: momentum_20d",
        kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
        factors=[{"factor_id": "momentum_20d"}],
    )
    return StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode="adhoc",
        strategy_spec=spec,
        factor_name="momentum_20d",
        start_date="20240101",
        end_date="20240331",
    )


def test_tree_fingerprint_changes_when_nested_file_changes(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    part = root / "part-000.parquet"
    part.write_bytes(b"first")

    first = fingerprint_path_tree(root)
    part.write_bytes(b"second-longer")
    second = fingerprint_path_tree(root)

    assert first != second


def test_cache_key_changes_with_complete_provenance(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    code_path = tmp_path / "generated.py"
    code_path.write_text("VALUE = 1\n", encoding="utf-8")
    factor_path = tmp_path / "custom_roe.py"
    factor_path.write_text("VALUE = 1\n", encoding="utf-8")
    factor_id = "custom_roe"
    strategy_tools._factor_registry(lake).save_factor(
        factor_id=factor_id,
        name="Custom ROE",
        version="0.1.0",
        implementation_ref=f"file:{factor_path}",
        required_columns=("symbol", "trade_date", "turnover", "roe"),
        lookback=1,
        params={},
        created_by="test",
    )
    spec = StrategySpec(
        strategy_id="provenance_strategy",
        name="Provenance strategy",
        kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
        factors=[{"factor_id": factor_id, "ascending": True}],
    )
    saved = SavedStrategy(
        strategy_id=spec.strategy_id,
        name=spec.name,
        version=spec.version,
        source=StrategySource.AGENT_GENERATED,
        status=ApprovalStatus.GENERATED_BY_LLM,
        spec=spec,
        implementation_ref=f"file:{code_path}",
        code_path=str(code_path),
    )
    config = StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode="inline",
        strategy_spec=spec,
        factor_name=factor_id,
        start_date="20240101",
        end_date="20240331",
        symbols=["000001.SZ"],
    )
    for dataset in (
        "tushare/daily",
        "tushare/trade_cal",
        "tushare/namechange",
        "tushare/daily_basic",
        "tushare/stock_basic",
        "tushare/index_weight",
        "tushare/fina_indicator",
    ):
        path = lake.dataset_path("raw", dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(dataset.encode())

    def key(universe: dict[str, object]) -> str:
        provenance = strategy_tools._backtest_provenance_manifest(
            lake,
            config=config,
            requested_factor_ids=[factor_id],
            saved_strategy=saved,
            effective_code_path=str(code_path),
            resolved_universe=universe,
        )
        return strategy_tools._backtest_cache_key(
            config=config,
            factor_name=factor_id,
            requested_factor_ids=[factor_id],
            provenance=provenance,
        )

    baseline = key({"symbols": ["000001.SZ"]})
    for dataset in (
        "tushare/trade_cal",
        "tushare/namechange",
        "tushare/daily_basic",
        "tushare/stock_basic",
        "tushare/index_weight",
        "tushare/fina_indicator",
    ):
        path = lake.dataset_path("raw", dataset)
        path.write_bytes(path.read_bytes() + b"-changed")
        changed = key({"symbols": ["000001.SZ"]})
        assert changed != baseline, dataset
        baseline = changed

    factor_path.write_text("VALUE = 2\n", encoding="utf-8")
    changed_factor = key({"symbols": ["000001.SZ"]})
    assert changed_factor != baseline
    baseline = changed_factor

    code_path.write_text("VALUE = 2\n", encoding="utf-8")
    changed_code = key({"symbols": ["000001.SZ"]})
    assert changed_code != baseline

    changed_universe = key({"symbols": ["000002.SZ"]})
    assert changed_universe != changed_code


def test_backtest_provenance_reuses_dataset_manifests(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    for dataset in (
        "tushare/daily",
        "tushare/fund_daily",
        "tushare/trade_cal",
        "tushare/suspend_d",
        "tushare/stk_limit",
        "tushare/namechange",
        "tushare/stock_basic",
        "tushare/index_weight",
        "tushare/index_member",
    ):
        lake.write_parquet(
            pd.DataFrame([{"dataset": dataset}]),
            "raw",
            dataset,
        )
    config = _fixture_config()
    strategy_tools._backtest_provenance_manifest(
        lake,
        config=config,
        requested_factor_ids=[],
        saved_strategy=None,
        effective_code_path=None,
        resolved_universe={"symbols": ["000001.SZ"]},
    )
    monkeypatch.setattr(
        dataset_manifests,
        "_content_digest",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("dataset payload must not be rehashed")
        ),
    )
    original_tree_fingerprint = strategy_tools.fingerprint_path_tree
    monkeypatch.setattr(
        strategy_tools,
        "fingerprint_path_tree",
        lambda path: (
            (_ for _ in ()).throw(
                AssertionError("datasets must use governed manifests")
            )
            if path.suffix == ".parquet"
            else original_tree_fingerprint(path)
        ),
    )

    strategy_tools._backtest_provenance_manifest(
        lake,
        config=config,
        requested_factor_ids=[],
        saved_strategy=None,
        effective_code_path=None,
        resolved_universe={"symbols": ["000001.SZ"]},
    )
