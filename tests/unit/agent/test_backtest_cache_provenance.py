from __future__ import annotations

from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.provenance import fingerprint_path_tree
from qmt_agent_trader.strategy.execution_adapter import StrategyBacktestConfig
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategyKind,
    StrategySource,
    StrategySpec,
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
    spec = StrategySpec(
        strategy_id="provenance_strategy",
        name="Provenance strategy",
        kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
        factors=[{"factor_id": "pb_rank", "ascending": True}],
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
        strategy_spec=spec,
        factor_name="pb_rank",
        start_date="20240101",
        end_date="20240331",
        symbols=["000001.SZ"],
    )
    for dataset in (
        "tushare/daily",
        "tushare/trade_cal",
        "tushare/namechange",
        "tushare/daily_basic",
    ):
        path = lake.dataset_path("raw", dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(dataset.encode())

    def key(universe: dict[str, object]) -> str:
        provenance = strategy_tools._backtest_provenance_manifest(
            lake,
            config=config,
            requested_factor_ids=["pb_rank"],
            saved_strategy=saved,
            effective_code_path=str(code_path),
            resolved_universe=universe,
        )
        return strategy_tools._backtest_cache_key(
            config=config,
            factor_name="pb_rank",
            requested_factor_ids=["pb_rank"],
            provenance=provenance,
        )

    baseline = key({"symbols": ["000001.SZ"]})
    for dataset in ("tushare/trade_cal", "tushare/namechange", "tushare/daily_basic"):
        path = lake.dataset_path("raw", dataset)
        path.write_bytes(path.read_bytes() + b"-changed")
        changed = key({"symbols": ["000001.SZ"]})
        assert changed != baseline
        baseline = changed

    code_path.write_text("VALUE = 2\n# changed\n", encoding="utf-8")
    changed_code = key({"symbols": ["000001.SZ"]})
    assert changed_code != baseline

    changed_universe = key({"symbols": ["000002.SZ"]})
    assert changed_universe != changed_code
