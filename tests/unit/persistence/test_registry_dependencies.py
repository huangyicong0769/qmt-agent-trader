from pathlib import Path

from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.factors.registry import FactorRegistry
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.paths import PersistencePaths
from qmt_agent_trader.strategy.registry import StrategyRegistry


def test_root_only_and_injected_registries_share_canonical_lock_namespace(
    tmp_path: Path,
) -> None:
    paths = PersistencePaths.from_settings(get_settings())
    injected_manager = LockManager(paths.locks_root)
    factor_root = tmp_path / "factors"
    strategy_root = tmp_path / "strategies"

    workflow_style = StrategyRegistry(strategy_root)
    agent_style = StrategyRegistry(
        strategy_root,
        lock_manager=injected_manager,
        atomic_store=AtomicFileStore(injected_manager),
    )
    research_style = FactorRegistry(factor_root)
    factor_tool_style = FactorRegistry(
        factor_root,
        lock_manager=injected_manager,
        atomic_store=AtomicFileStore(injected_manager),
    )

    assert workflow_style.lock_manager.locks_root == paths.locks_root
    assert research_style.lock_manager.locks_root == paths.locks_root
    assert workflow_style.lock_manager.lock_path_for_resource(
        workflow_style.registry_path
    ) == agent_style.lock_manager.lock_path_for_resource(agent_style.registry_path)
    assert research_style.lock_manager.lock_path_for_resource(
        research_style.registry_path
    ) == factor_tool_style.lock_manager.lock_path_for_resource(
        factor_tool_style.registry_path
    )
