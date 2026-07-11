from pathlib import Path

from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.persistence.catalog import StoreCatalog
from qmt_agent_trader.persistence.paths import PersistencePaths


def test_catalog_names_every_real_store_and_excludes_operational_roots(tmp_path: Path) -> None:
    paths = PersistencePaths.from_settings(Settings(project_root=tmp_path))
    catalog = StoreCatalog.canonical(paths)
    by_name = {store.name: store for store in catalog.stores}
    expected = {
        "control_db",
        "lake_raw",
        "lake_silver",
        "lake_gold",
        "lake_metadata",
        "factor_registry",
        "strategy_registry",
        "todos",
        "experiments",
        "sessions",
        "universes",
        "legacy_universes",
        "approvals",
        "order_plans",
        "reports",
        "audit",
        "generated_code",
    }
    assert set(by_name) == expected
    assert by_name["factor_registry"].path == paths.data_root / "factors/registry.json"
    assert by_name["universes"].path == paths.registries_root / "universes"
    assert by_name["legacy_universes"].path == paths.data_root / "universes/registry"
    excluded = {paths.cache_root, paths.locks_root, paths.backup_root, paths.quarantine_root}
    assert not excluded & {store.path for store in catalog.stores}
    assert all(store.owner and store.lock_resource and store.backup for store in catalog.stores)
