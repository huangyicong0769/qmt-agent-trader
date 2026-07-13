"""Command line interface."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Annotated, cast

import pandas as pd
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from qmt_agent_trader.agent.audit import scrub_sensitive
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.agent.permissions import ToolCallMode
from qmt_agent_trader.agent.runtime import AgentRuntime, build_default_runtime
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.workflows.factor_discovery import (
    FactorDiscoveryWorkflow,
    run_factor_discovery,
)
from qmt_agent_trader.agent.workflows.self_bootstrap import SelfBootstrapWorkflow
from qmt_agent_trader.agent.workflows.strategy_discovery import run_strategy_discovery
from qmt_agent_trader.agent.workflows.strategy_engineering import (
    StrategyEngineeringWorkflow,
)
from qmt_agent_trader.backtest.service import compare_backtest_reports, run_backtest_report
from qmt_agent_trader.broker.order_plan import OrderPlan
from qmt_agent_trader.broker.remote_client import RemoteQMTBrokerClient
from qmt_agent_trader.broker.risk import run_order_plan_risk_checks
from qmt_agent_trader.cli.todo_render import empty_todo_state, render_todo_panel
from qmt_agent_trader.cli.tui import run_tui
from qmt_agent_trader.core.audit import AuditLogger
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.core.types import ApprovalStatus, RiskStatus
from qmt_agent_trader.data.macro import MACRO_DATASETS
from qmt_agent_trader.data.providers.base import FetchItem
from qmt_agent_trader.data.providers.tushare.client import TushareClient as GenericTushareClient
from qmt_agent_trader.data.providers.tushare.fetcher import TushareFetcher
from qmt_agent_trader.data.providers.tushare.ledger_migration import (
    repair_tushare_usage_ledger,
)
from qmt_agent_trader.data.providers.tushare.planner import TusharePlannerConfig
from qmt_agent_trader.data.providers.tushare.provider import TushareProvider
from qmt_agent_trader.data.providers.tushare.quota import (
    QuotaSource,
    TushareQuotaManager,
    TushareUsageLedger,
    profile_from_settings,
)
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.table_builder import DataTableBuilder
from qmt_agent_trader.factors.service import compute_factor_to_lake, validate_factor
from qmt_agent_trader.persistence.artifacts import ArtifactStore, artifact_store_for_root
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import StorageError
from qmt_agent_trader.persistence.initialization import initialize_persistence
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.operations import StorageOperations, as_json
from qmt_agent_trader.persistence.paths import PersistencePaths
from qmt_agent_trader.services.order_plan_service import (
    OrderPlanEvent,
    append_order_plan_event,
    build_sample_paper_order_plan,
    load_order_plan,
    load_order_plan_events,
    save_order_plan,
)
from qmt_agent_trader.strategy.approval import StrategyApproval, write_approval_file
from qmt_agent_trader.strategy.models import StrategySource
from qmt_agent_trader.strategy.registry import StrategyRegistry

app = typer.Typer(help="Mac control plane for QMT agent trading.")
data_app = typer.Typer(help="Data lake commands.")
factor_app = typer.Typer(help="Factor commands.")
backtest_app = typer.Typer(help="Backtest commands.")
agent_app = typer.Typer(help="LLM research agent commands.")
strategy_app = typer.Typer(help="Strategy approval commands.")
broker_app = typer.Typer(help="Remote QMT broker commands.")
trade_app = typer.Typer(help="Order plan and trading commands.")
storage_app = typer.Typer(help="Local persistence health and operations.")

app.add_typer(data_app, name="data")
app.add_typer(factor_app, name="factor")
app.add_typer(backtest_app, name="backtest")
app.add_typer(agent_app, name="agent")
app.add_typer(strategy_app, name="strategy")
app.add_typer(broker_app, name="broker")
app.add_typer(trade_app, name="trade")
app.add_typer(storage_app, name="storage")


def _settings() -> Settings:
    return get_settings()


def _storage_operations() -> StorageOperations:
    settings = _settings()
    return StorageOperations(
        PersistencePaths.from_settings(settings),
        timeout_seconds=settings.remote_data_lock_timeout_seconds,
    )


@storage_app.command("inventory")
def storage_inventory() -> None:
    try:
        print_json([as_json(item) for item in _storage_operations().inventory()])
    except StorageError as exc:
        _raise_storage_cli_error(exc)


@storage_app.command("verify")
def storage_verify(deep: bool = typer.Option(False, "--deep")) -> None:
    try:
        result = _storage_operations().verify(deep=deep)
    except StorageError as exc:
        _raise_storage_cli_error(exc)
    print_json(as_json(result))
    if not result.healthy:
        raise typer.Exit(code=1)


@storage_app.command("migrate")
def storage_migrate(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    try:
        applied = _storage_operations().migrate(dry_run=dry_run)
    except StorageError as exc:
        _raise_storage_cli_error(exc)
    print_json({"status": "ok", "dry_run": dry_run, "migrations": applied})


@storage_app.command("backup")
def storage_backup() -> None:
    try:
        receipt = _storage_operations().backup()
    except StorageError as exc:
        _raise_storage_cli_error(exc)
    print_json({"status": "ok", **as_json(receipt)})


@storage_app.command("locks")
def storage_locks() -> None:
    try:
        print_json(_storage_operations().locks_report())
    except StorageError as exc:
        _raise_storage_cli_error(exc)


@storage_app.command("quarantine")
def storage_quarantine(store: str, record: str) -> None:
    try:
        receipt = _storage_operations().quarantine(store, record)
    except StorageError as exc:
        _raise_storage_cli_error(exc)
    print_json({"status": "quarantined", **as_json(receipt)})


def _raise_storage_cli_error(exc: StorageError) -> None:
    print_json(
        scrub_sensitive({"status": "error", "error_type": type(exc).__name__, "reason": exc.reason})
    )
    raise typer.Exit(code=1) from exc


def _artifact_store(root: Path) -> ArtifactStore:
    settings = _settings()
    paths = PersistencePaths.from_settings(settings)
    return artifact_store_for_root(
        root,
        lock_manager=LockManager(
            paths.locks_root,
            timeout_seconds=settings.remote_data_lock_timeout_seconds,
        ),
    )


def _strategy_registry() -> StrategyRegistry:
    settings = _settings()
    paths = PersistencePaths.from_settings(settings)
    lock_manager = LockManager(
        paths.locks_root,
        timeout_seconds=settings.remote_data_lock_timeout_seconds,
    )
    return StrategyRegistry(
        paths.data_root / "strategies",
        lock_manager=lock_manager,
        atomic_store=AtomicFileStore(lock_manager),
    )


def _broker_client() -> RemoteQMTBrokerClient:
    settings = _settings()
    api_key = (
        settings.qmt_gateway_api_key.get_secret_value() if settings.qmt_gateway_api_key else ""
    )
    secret = (
        settings.qmt_gateway_hmac_secret.get_secret_value()
        if settings.qmt_gateway_hmac_secret
        else ""
    )
    if not api_key or not secret:
        raise typer.BadParameter("QMT gateway API key and HMAC secret must be configured in .env")
    return RemoteQMTBrokerClient(
        base_url=settings.qmt_gateway_base_url,
        api_key=api_key,
        hmac_secret=secret,
    )


def _generic_tushare_client() -> GenericTushareClient:
    settings = _settings()
    token = settings.tushare_token.get_secret_value() if settings.tushare_token else None
    return GenericTushareClient(
        token=token,
        timeout_seconds=settings.remote_data_http_timeout_seconds,
    )


def _data_lake() -> DataLake:
    settings = _settings()
    paths = PersistencePaths.from_settings(settings)
    lock_manager = LockManager(
        paths.locks_root, timeout_seconds=settings.remote_data_lock_timeout_seconds
    )
    lake = DataLake(
        root=paths.lake_root,
        duckdb_path=paths.control_db_path,
        parquet_lock_timeout_seconds=settings.remote_data_lock_timeout_seconds,
        database_coordinator=DatabaseCoordinator(paths.control_db_path, lock_manager),
        lock_manager=lock_manager,
    )
    initialize_persistence(lake, raise_on_legacy_error=False)
    return lake


def _tushare_provider() -> TushareProvider:
    settings = _settings()
    lake = _data_lake()
    initialize_persistence(lake, raise_on_legacy_error=False)
    quota_profile = profile_from_settings(
        source=cast(QuotaSource, settings.tushare_quota_profile_source),
        points=settings.tushare_points,
        max_requests_per_minute=settings.tushare_max_requests_per_minute,
        max_requests_per_day_per_api=settings.tushare_max_requests_per_day_per_api,
    )
    ledger = TushareUsageLedger.from_data_lake(
        lake,
        lock_timeout_seconds=settings.remote_data_lock_timeout_seconds,
    )
    quota_manager = TushareQuotaManager(profile=quota_profile, ledger=ledger)
    return TushareProvider(
        fetcher=TushareFetcher(
            _generic_tushare_client(),
            lake,
            min_interval_seconds=settings.remote_data_min_interval_seconds,
            retry_attempts=settings.remote_data_retry_attempts,
            retry_backoff_seconds=settings.remote_data_retry_backoff_seconds,
            quota_manager=quota_manager,
            usage_ledger=ledger,
        ),
        planner_config=TusharePlannerConfig(
            symbol_fanout_threshold=30,
            quota_profile=quota_profile,
            max_days_per_batch=settings.remote_data_max_days_per_call,
        ),
        usage_ledger=ledger,
    )


def _audit_logger(name: str) -> AuditLogger:
    settings = _settings()
    paths = PersistencePaths.from_settings(settings)
    lock_manager = LockManager(
        paths.locks_root, timeout_seconds=settings.remote_data_lock_timeout_seconds
    )
    return AuditLogger(
        paths.audit_root / f"{name}.jsonl",
        atomic_store=AtomicFileStore(lock_manager),
        fsync=settings.audit_fsync,
        rotation_bytes=settings.audit_rotation_bytes,
    )


@app.command()
def tui() -> None:
    """Run the Textual TUI skeleton."""
    run_tui()


@app.command("web")
def serve_web(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(7860),
    reload: bool = typer.Option(False),
) -> None:
    """Start QMT Agent Studio web interface."""
    import uvicorn

    from qmt_agent_trader.web.app import create_app

    web_app = create_app()
    uvicorn.run(web_app, host=host, port=port, reload=reload)


@data_app.command("capabilities")
def data_capabilities(
    category: Annotated[str | None, typer.Option("--category")] = None,
    asset_type: Annotated[str | None, typer.Option("--asset-type")] = None,
) -> None:
    """List registry-driven Tushare endpoint capabilities."""
    capability = TushareProvider().list_capabilities(category=category, asset_type=asset_type)
    print_json({"status": "OK", "source": capability.source, "endpoints": capability.endpoints})


@data_app.command("plan-fetch")
def data_plan_fetch(
    api: Annotated[str, typer.Option("--api")],
    symbols: Annotated[str | None, typer.Option("--symbols")] = None,
    fields: Annotated[str | None, typer.Option("--fields")] = None,
    from_date: Annotated[str | None, typer.Option("--from")] = None,
    to_date: Annotated[str | None, typer.Option("--to")] = None,
    trade_date: Annotated[str | None, typer.Option("--trade-date")] = None,
) -> None:
    """Plan a registry-validated Tushare fetch without contacting Tushare."""
    item = FetchItem(
        api_name=api,
        symbols=_csv_values(symbols),
        fields=_csv_values(fields) or None,
        start_date=from_date,
        end_date=to_date,
        trade_date=trade_date,
    )
    print_json(_tushare_provider().plan_fetch([item]).as_dict())


@data_app.command("fetch")
def data_fetch(
    plan: Annotated[Path | None, typer.Option("--plan")] = None,
    api: Annotated[str | None, typer.Option("--api")] = None,
    symbols: Annotated[str | None, typer.Option("--symbols")] = None,
    fields: Annotated[str | None, typer.Option("--fields")] = None,
    from_date: Annotated[str | None, typer.Option("--from")] = None,
    to_date: Annotated[str | None, typer.Option("--to")] = None,
    execute_plan: Annotated[bool, typer.Option("--execute-plan")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Execute a registry-validated Tushare fetch plan."""
    provider = _tushare_provider()
    if plan is not None:
        payload = json.loads(plan.read_text(encoding="utf-8"))
        items = [
            FetchItem(
                api_name=str(item["api_name"]),
                symbols=[str(value) for value in item.get("symbols", [])],
                fields=[str(value) for value in item.get("fields", [])] or None,
                start_date=item.get("start_date"),
                end_date=item.get("end_date"),
                trade_date=item.get("trade_date"),
                params=dict(item.get("params", {})),
            )
            for item in payload.get("items", [])
        ]
    else:
        if api is None:
            raise typer.BadParameter("--api is required when --plan is not provided")
        items = [
            FetchItem(
                api_name=api,
                symbols=_csv_values(symbols),
                fields=_csv_values(fields) or None,
                start_date=from_date,
                end_date=to_date,
            )
        ]
    fetch_plan = provider.plan_fetch(items)
    result = provider.run_fetch(fetch_plan, execute_plan=execute_plan, dry_run=dry_run)
    print_json({**result.as_dict(), "plan": fetch_plan.as_dict()})


@data_app.command("build-table")
def data_build_table(
    table: Annotated[str, typer.Option("--table")],
    snapshot_as_of_date: Annotated[str | None, typer.Option("--snapshot-as-of-date")] = None,
) -> None:
    """Build one allowed silver table from registry-driven raw datasets."""
    print_json(DataTableBuilder(_data_lake()).build(table, snapshot_as_of_date=snapshot_as_of_date))


@data_app.command("validate")
def data_validate() -> None:
    """Validate local data lake artifacts."""
    lake = _data_lake()
    expected = [
        lake.dataset_path("raw", "tushare/trade_cal"),
        lake.dataset_path("raw", "tushare/stock_basic"),
        lake.dataset_path("raw", "tushare/fund_basic"),
    ]
    missing = [str(path) for path in expected if not path.exists()]
    print_json(
        {
            "status": "ok" if not missing else "missing_data",
            "missing": missing,
            "duckdb_exists": lake.duckdb_path.exists(),
        }
    )


@data_app.command("repair-tushare-ledger")
def data_repair_tushare_ledger(
    quarantine_corrupt: Annotated[
        bool,
        typer.Option("--quarantine-corrupt"),
    ] = False,
) -> None:
    """Inspect the legacy usage ledger and explicitly quarantine it if corrupt."""
    lake = _data_lake()
    initialize_persistence(lake, migrate_legacy_ledger=False)
    ledger = TushareUsageLedger.from_data_lake(
        lake,
        lock_timeout_seconds=_settings().remote_data_lock_timeout_seconds,
    )
    typer.echo(
        json.dumps(
            repair_tushare_usage_ledger(
                ledger,
                quarantine_corrupt=quarantine_corrupt,
            ),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


@data_app.command("validate-fundamentals")
def data_validate_fundamentals() -> None:
    """Validate local fundamentals datasets and PIT columns."""
    lake = _data_lake()
    specs = {
        "tushare/daily_basic": {
            "required_columns": ["ts_code", "trade_date"],
            "pit_columns": ["trade_date"],
            "pit_safe": True,
        },
        "tushare/income": {
            "required_columns": ["ts_code", "end_date", "ann_date"],
            "pit_columns": ["ann_date", "f_ann_date"],
            "pit_safe": True,
        },
        "tushare/balancesheet": {
            "required_columns": ["ts_code", "end_date", "ann_date"],
            "pit_columns": ["ann_date", "f_ann_date"],
            "pit_safe": True,
        },
        "tushare/cashflow": {
            "required_columns": ["ts_code", "end_date", "ann_date"],
            "pit_columns": ["ann_date", "f_ann_date"],
            "pit_safe": True,
        },
        "tushare/fina_indicator": {
            "required_columns": ["ts_code", "end_date", "ann_date"],
            "pit_columns": ["ann_date"],
            "pit_safe": True,
        },
    }
    print_json({"status": "ok", "datasets": _dataset_validation(lake, specs)})


@data_app.command("validate-macro")
def data_validate_macro() -> None:
    """Validate local macro datasets and visibility metadata."""
    lake = _data_lake()
    specs = {
        spec.raw_dataset: {
            "required_columns": spec.key_columns,
            "pit_columns": [spec.date_column],
            "pit_safe": spec.pit_safe,
            "visibility_rule": spec.visibility_rule,
        }
        for spec in MACRO_DATASETS.values()
    }
    print_json({"status": "ok", "datasets": _dataset_validation(lake, specs)})


@data_app.command("migrate-new-layout")
def data_migrate_new_layout(
    keep_legacy: Annotated[bool, typer.Option("--keep-legacy")] = False,
) -> None:
    """Migrate stable old raw Tushare files into raw/tushare/*.parquet layout."""
    lake = _data_lake()
    mapping = {
        "tushare_daily": ("tushare/daily", ["ts_code", "trade_date"]),
        "tushare_daily_basic": ("tushare/daily_basic", ["ts_code", "trade_date"]),
        "tushare_fund_daily": ("tushare/fund_daily", ["ts_code", "trade_date"]),
        "tushare_stock_basic": ("tushare/stock_basic", ["ts_code"]),
        "tushare_etf_basic": ("tushare/fund_basic", ["ts_code"]),
        "tushare_index_basic": ("tushare/index_basic", ["ts_code"]),
        "tushare_index_daily": ("tushare/index_daily", ["ts_code", "trade_date"]),
        "tushare_trade_calendar": ("tushare/trade_cal", ["exchange", "cal_date"]),
        "tushare_namechange": ("tushare/namechange", ["ts_code", "start_date", "name"]),
        "tushare_income": ("tushare/income", ["ts_code", "end_date", "ann_date", "report_type"]),
        "tushare_balancesheet": (
            "tushare/balancesheet",
            ["ts_code", "end_date", "ann_date", "report_type"],
        ),
        "tushare_cashflow": (
            "tushare/cashflow",
            ["ts_code", "end_date", "ann_date", "report_type"],
        ),
        "tushare_fina_indicator": (
            "tushare/fina_indicator",
            ["ts_code", "end_date", "ann_date"],
        ),
        "tushare_dividend": ("tushare/dividend", ["ts_code", "end_date", "ann_date", "div_proc"]),
        "tushare_macro_cn_gdp": ("tushare/cn_gdp", ["quarter"]),
        "tushare_macro_cn_cpi": ("tushare/cn_cpi", ["month"]),
        "tushare_macro_cn_ppi": ("tushare/cn_ppi", ["month"]),
        "tushare_macro_shibor": ("tushare/shibor", ["date"]),
        "tushare_suspend": ("tushare/suspend_d", ["ts_code", "trade_date"]),
        "tushare_stk_limit": ("tushare/stk_limit", ["ts_code", "trade_date"]),
    }
    migrations: list[dict[str, object]] = []
    for old_name, (new_name, keys) in mapping.items():
        source_names = _legacy_raw_sources_for_new_layout(lake, old_name)
        if not source_names:
            continue
        frames = []
        for source_name in source_names:
            frame = lake.read_parquet("raw", source_name)
            if "_empty" in frame.columns:
                frame = frame.drop(columns=["_empty"])
            if not frame.empty:
                frames.append(frame)
        frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            migrations.append(
                {
                    "legacy_sources": source_names,
                    "new_name": new_name,
                    "status": "SKIPPED",
                    "missing_key_columns": missing,
                }
            )
            continue
        path = lake.write_incremental_parquet(frame, "raw", new_name, key_columns=keys)
        if not keep_legacy:
            for source_name in source_names:
                source_path = lake.dataset_path("raw", source_name)
                if source_path.exists():
                    source_path.unlink()
        migrations.append(
            {
                "legacy_sources": source_names,
                "new_name": new_name,
                "status": "migrated",
                "path": str(path),
                "rows": len(frame),
                "legacy_removed": not keep_legacy,
            }
        )
    print_json({"status": "ok", "migrations": migrations})


def _legacy_raw_sources_for_new_layout(lake: DataLake, stable_name: str) -> list[str]:
    names: list[str] = []
    if lake.dataset_path("raw", stable_name).exists():
        names.append(stable_name)
    batch_pattern = re.compile(rf"^{re.escape(stable_name)}_\d{{8}}_\d{{8}}$")
    names.extend(
        name
        for name in lake.list_dataset_names("raw", prefix=f"{stable_name}_")
        if batch_pattern.match(name)
    )
    return sorted(dict.fromkeys(names))


@data_app.command("qmt-sync")
def data_qmt_sync(
    from_date: Annotated[str, typer.Option("--from")],
    to_date: Annotated[str, typer.Option("--to")],
) -> None:
    """Plan QMT local bar sync."""
    print_json({"status": "planned", "from": from_date, "to": to_date})


@factor_app.command("compute")
def factor_compute(
    name: Annotated[str, typer.Option("--name")],
    date: Annotated[str, typer.Option("--date")],
) -> None:
    try:
        print_json(compute_factor_to_lake(_data_lake(), name=name, date=date).as_dict())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@factor_app.command("validate")
def factor_validate(
    name: Annotated[str, typer.Option("--name")],
    start: Annotated[str, typer.Option("--start")],
    end: Annotated[str, typer.Option("--end")],
) -> None:
    try:
        print_json(validate_factor(_data_lake(), name=name, start=start, end=end).as_dict())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@backtest_app.command("run")
def backtest_run(
    config: Path = Path("configs/backtest.yaml"),
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    signal_date: Annotated[str | None, typer.Option("--signal-date")] = None,
    quantity: Annotated[int, typer.Option("--quantity")] = 100,
) -> None:
    try:
        summary = run_backtest_report(
            _data_lake(),
            reports_dir=PersistencePaths.from_settings(_settings()).reports_root / "backtests",
            symbol=symbol,
            signal_date=signal_date,
            quantity=quantity,
            config_path=str(config),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = summary.as_dict()
    payload["config"] = str(config)
    payload["mode"] = "daily_t_plus_1"
    print_json(payload)


@backtest_app.command("compare")
def backtest_compare(runs: str = "latest-10") -> None:
    lake = _data_lake()
    print_json(
        compare_backtest_reports(
            PersistencePaths.from_settings(_settings()).reports_root / "backtests",
            limit=_parse_latest_limit(runs),
            lock_manager=lake.lock_manager,
        )
    )


@agent_app.command("discover-factors")
def discover_factors(theme: Annotated[str, typer.Option("--theme")]) -> None:
    print_json(run_factor_discovery(theme, settings=_settings()))


@agent_app.command("discover-strategies")
def discover_strategies(universe: Annotated[str, typer.Option("--universe")]) -> None:
    print_json(run_strategy_discovery(universe, settings=_settings()))


@agent_app.command("tools")
def agent_tools() -> None:
    runtime = build_default_runtime(_settings())
    print_json({"tools": runtime.list_tools(agent_callable_only=True)})


@agent_app.command("call-tool")
def agent_call_tool(
    name: Annotated[str, typer.Option("--name")],
    params: Annotated[str, typer.Option("--params")] = "{}",
) -> None:
    payload = _parse_json_params(params)
    if not isinstance(payload, dict):
        raise typer.BadParameter("--params must be a JSON object")
    runtime = _agent_runtime()
    result = runtime.run_tool(
        name,
        payload,
        ToolContext(
            run_id="cli-call-tool",
            requested_by_llm=True,
            call_mode=ToolCallMode.AUTONOMOUS_AGENT,
            dry_run=False,
        ),
    )
    print_json({"tool": name, "result": result})


@agent_app.command("ask")
def agent_ask(
    prompt: Annotated[str, typer.Option("--prompt")],
    max_rounds: Annotated[int, typer.Option("--max-rounds")] = 100,
    session_id: Annotated[str, typer.Option("--session-id")] = "cli-default",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    if json_output:
        runtime = build_default_runtime(_settings())
        try:
            result = runtime.ask(prompt, max_rounds=max_rounds, session_id=session_id)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        print_json(
            {
                "content": result.content,
                "tool_calls": [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": call.result,
                    }
                    for call in result.tool_calls
                ],
            }
        )
        return

    asyncio.run(
        _agent_ask_stream(
            prompt=prompt,
            max_rounds=max_rounds,
            session_id=session_id,
        )
    )


async def _agent_ask_stream(*, prompt: str, max_rounds: int, session_id: str) -> None:
    runtime = build_default_runtime(_settings())
    orchestrator = AgentOrchestrator(runtime=runtime)
    console = Console()
    todo_state = empty_todo_state()
    event_lines: list[str] = []
    answer_parts: list[str] = []
    final_answer = ""

    def view() -> Group:
        answer = final_answer or "".join(answer_parts) or "Waiting for assistant output..."
        events = "\n".join(event_lines[-10:]) or "No tool calls yet."
        return Group(
            render_todo_panel(todo_state),
            Panel(events, title="Events", border_style="green"),
            Panel(answer, title="Assistant", border_style="cyan"),
        )

    with Live(view(), console=console, refresh_per_second=8) as live:
        async for event in orchestrator.execute_stream(
            prompt,
            session_id=session_id,
            max_rounds=max_rounds,
        ):
            if event.type == "todo_status":
                todo_state = event.data
            elif event.type == "tool_start":
                event_lines.append(f"Calling {event.data.get('tool_name', '')}")
            elif event.type == "tool_done":
                event_lines.append(f"Done {event.data.get('tool_name', '')}")
            elif event.type == "token":
                answer_parts.append(event.message)
            elif event.type == "final_message":
                final_answer = event.message
            elif event.type == "error":
                event_lines.append(f"Error: {event.message}")
            elif event.type == "done":
                event_lines.append(event.message)
            live.update(view())


def _agent_registry() -> AgentToolRegistry:
    return _agent_runtime().agent_registry()


def _agent_runtime() -> AgentRuntime:
    settings = _settings()
    return build_default_runtime(settings)


def _agent_store() -> ExperimentStore:
    settings = _settings()
    paths = PersistencePaths.from_settings(settings)
    return ExperimentStore(
        paths.experiments_root,
        locks_root=paths.locks_root,
        quarantine_root=paths.quarantine_root / "experiments",
    )


@agent_app.command("experiments")
def agent_experiments(
    query: Annotated[str | None, typer.Option("--query")] = None,
    tag: Annotated[str | None, typer.Option("--tag")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """List / search recent agent experiments."""
    store = _agent_store()
    tags = [tag] if tag else None
    results = store.search_experiments(query=query, tags=tags, limit=limit)
    print_json(
        {
            "count": len(results),
            "experiments": [r.model_dump(mode="json") for r in results],
        }
    )


@agent_app.command("experiment")
def agent_experiment(experiment_id: Annotated[str, typer.Option("--id")]) -> None:
    """Show one experiment by id."""
    store = _agent_store()
    try:
        exp = store.get_experiment(experiment_id)
        print_json(exp.model_dump(mode="json"))
    except Exception as exc:
        print_json({"error": str(exc)})


@agent_app.command("run-factor-discovery")
def agent_run_factor_discovery(
    theme: Annotated[str, typer.Option("--theme")],
    universe: Annotated[str, typer.Option("--universe")] = "stock_etf",
    start: Annotated[str, typer.Option("--start")] = "20200101",
    end: Annotated[str, typer.Option("--end")] = "20260624",
) -> None:
    """Run the new tool-chain factor discovery pipeline."""
    reg = _agent_registry()
    store = _agent_store()
    workflow = FactorDiscoveryWorkflow(reg, store)
    exp = workflow.run(theme, universe, start, end)
    print_json(exp.model_dump(mode="json"))


@agent_app.command("write-strategy")
def agent_write_strategy(
    idea: Annotated[str, typer.Option("--idea")],
    factors: Annotated[str, typer.Option("--factors")],
    universe: Annotated[str, typer.Option("--universe")] = "stock_etf",
    start: Annotated[str, typer.Option("--start")] = "20200101",
    end: Annotated[str, typer.Option("--end")] = "20260624",
) -> None:
    """Run the strategy engineering pipeline."""
    reg = _agent_registry()
    store = _agent_store()
    factor_list = [f.strip() for f in factors.split(",") if f.strip()]
    workflow = StrategyEngineeringWorkflow(reg, store)
    exp = workflow.run(idea, factor_list, universe, start, end)
    print_json(exp.model_dump(mode="json"))


@agent_app.command("self-bootstrap")
def agent_self_bootstrap(
    recent: Annotated[int, typer.Option("--recent")] = 10,
    experiment_ids: Annotated[str | None, typer.Option("--experiment-ids")] = None,
) -> None:
    """Run the self-bootstrap pipeline to detect tool gaps."""
    reg = _agent_registry()
    store = _agent_store()
    ids = [i.strip() for i in experiment_ids.split(",")] if experiment_ids else [f"auto_{recent}"]
    workflow = SelfBootstrapWorkflow(reg, store)
    exp = workflow.run(ids)
    print_json(exp.model_dump(mode="json"))


@strategy_app.command("list")
def strategy_list() -> None:
    registry = _strategy_registry()
    print_json(
        {
            "strategies": [
                item.model_dump(mode="json")
                for item in registry.list_strategies(include_builtins=True)
            ]
        }
    )


@strategy_app.command("candidates")
def strategy_candidates(query: Annotated[str | None, typer.Option("--query")] = None) -> None:
    registry = _agent_registry()
    context = ToolContext(run_id="cli_strategy_candidates", requested_by_llm=False)
    print_json(registry.run_tool("list_strategy_candidates", {"query": query or ""}, context))


@strategy_app.command("show")
def strategy_show(strategy_id: Annotated[str, typer.Option("--strategy-id")]) -> None:
    saved = _strategy_registry().get_strategy(strategy_id)
    if saved is None:
        raise typer.BadParameter(f"strategy not found: {strategy_id}")
    print_json(saved.model_dump(mode="json"))


@strategy_app.command("review")
def strategy_review(strategy_id: Annotated[str, typer.Option("--strategy-id")]) -> None:
    print_json({"strategy_id": strategy_id, "status": ApprovalStatus.REVIEW_REQUIRED})


@strategy_app.command("approve")
def strategy_approve(
    strategy_id: Annotated[str, typer.Option("--strategy-id")],
    paper_only: bool = True,
) -> None:
    registry = _strategy_registry()
    saved = registry.get_strategy(strategy_id)
    if saved is None:
        raise typer.BadParameter(f"strategy not found in registry: {strategy_id}")
    if saved.source == StrategySource.BUILTIN:
        raise typer.BadParameter("built-in strategies cannot be approved from CLI")
    if saved.status != ApprovalStatus.REVIEW_REQUIRED:
        raise typer.BadParameter("strategy must be REVIEW_REQUIRED before approval")
    approval = StrategyApproval(
        strategy_id=strategy_id,
        strategy_name=saved.name,
        strategy_version=saved.version,
        approved_by="human",
        allowed_universe=["A_SHARE_STOCK", "ETF"],
        allowed_accounts=["paper_account"],
        max_single_position_pct=saved.spec.portfolio.max_single_position_pct,
        max_turnover_daily_pct=0.30,
        max_order_value=100000,
        live_trading_allowed=False,
        paper_trading_allowed=True,
        notes="First approval for paper trading only.",
    )
    paths = PersistencePaths.from_settings(_settings())
    path = write_approval_file(
        approval,
        artifact_store=_artifact_store(paths.approvals_root),
    )
    registry.attach_approval(strategy_id, str(path))
    registry.update_status(strategy_id, ApprovalStatus.APPROVED, trusted=True)
    print_json({"status": "APPROVED", "paper_only": paper_only, "path": str(path)})


@strategy_app.command("retire")
def strategy_retire(strategy_id: Annotated[str, typer.Option("--strategy-id")]) -> None:
    print_json({"strategy_id": strategy_id, "status": "RETIRE_REQUEST_RECORDED"})


@broker_app.command("health")
def broker_health() -> None:
    print_json(_broker_client().health())


@broker_app.command("positions")
def broker_positions() -> None:
    print_json(_broker_client().query_positions())


@broker_app.command("asset")
def broker_asset() -> None:
    print_json(_broker_client().query_asset())


@trade_app.command("generate-plan")
def trade_generate_plan(strategy_id: Annotated[str, typer.Option("--strategy-id")]) -> None:
    plan = build_sample_paper_order_plan(strategy_id)
    paths = PersistencePaths.from_settings(_settings())
    path = save_order_plan(
        plan,
        artifact_store=_artifact_store(paths.order_plans_root),
    )
    print_json({"status": "generated", "path": str(path), "plan_hash": plan.plan_hash})


@trade_app.command("risk-check")
def trade_risk_check(plan: Annotated[str, typer.Option("--plan")]) -> None:
    order_plan = _load_plan_or_error(plan)
    plans_root = PersistencePaths.from_settings(_settings()).order_plans_root
    artifact_store = _artifact_store(plans_root)
    _load_plan_events_or_error(order_plan.order_plan_id, artifact_store=artifact_store)
    result = run_order_plan_risk_checks(order_plan)
    payload = {
        "plan": plan,
        "order_plan_id": order_plan.order_plan_id,
        "status": result.status,
        "checks": [check.model_dump(mode="json") for check in result.checks],
    }
    _audit_logger("trade").append("trade.risk_check", "cli", payload)
    append_order_plan_event(
        order_plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="cli",
        details={"status": result.status.value},
        artifact_store=artifact_store,
    )
    print_json(payload)


@trade_app.command("paper")
def trade_paper(plan: Annotated[str, typer.Option("--plan")]) -> None:
    order_plan = _load_plan_or_error(plan)
    plans_root = PersistencePaths.from_settings(_settings()).order_plans_root
    artifact_store = _artifact_store(plans_root)
    event_history = _load_plan_events_or_error(
        order_plan.order_plan_id,
        artifact_store=artifact_store,
    )
    result = run_order_plan_risk_checks(order_plan)
    if result.status != RiskStatus.PASSED:
        raise typer.BadParameter("risk checks failed")
    try:
        order_plan.assert_submittable(live=False)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        "plan": plan,
        "order_plan_id": order_plan.order_plan_id,
        "status": "PAPER_ACCEPTED",
        "live": False,
        "idempotency_key": order_plan.idempotency_key,
        "plan_hash": order_plan.plan_hash,
        "event_history_count": len(event_history),
    }
    _audit_logger("trade").append("trade.paper", "cli", payload)
    append_order_plan_event(
        order_plan.order_plan_id,
        event_type="PAPER_ACCEPTED",
        actor="cli",
        details={"live": False, "idempotency_key": order_plan.idempotency_key},
        artifact_store=artifact_store,
    )
    print_json(payload)


@trade_app.command("submit")
def trade_submit(
    plan: Annotated[str, typer.Option("--plan")],
    confirm_live: bool = False,
) -> None:
    settings = _settings()
    if not settings.live_trading_enabled or not confirm_live:
        raise typer.BadParameter(
            "live submit refused unless config allows live and --confirm-live is provided"
        )
    print_json({"plan": plan, "status": "SUBMIT_READY"})


def _csv_values(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_validation(
    lake: DataLake,
    specs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name, spec in specs.items():
        path = lake.dataset_path("raw", name)
        required_columns = _string_list(spec.get("required_columns", []))
        pit_columns = _string_list(spec.get("pit_columns", []))
        if not path.exists():
            results.append(
                {
                    "dataset": name,
                    "exists": False,
                    "row_count": 0,
                    "date_range": None,
                    "required_columns": required_columns,
                    "pit_columns": pit_columns,
                    "missing_required_columns": required_columns,
                    "pit_safe": bool(spec.get("pit_safe", False)),
                    **{
                        key: value
                        for key, value in spec.items()
                        if key not in {"required_columns", "pit_columns", "pit_safe"}
                    },
                }
            )
            continue
        frame = lake.read_parquet("raw", name)
        missing = [column for column in required_columns if column not in frame.columns]
        date_column = next((column for column in pit_columns if column in frame.columns), None)
        date_range = None
        if date_column is not None and not frame.empty:
            values = frame[date_column].dropna().astype(str)
            if not values.empty:
                date_range = {"start": values.min(), "end": values.max()}
        results.append(
            {
                "dataset": name,
                "exists": True,
                "row_count": len(frame),
                "date_range": date_range,
                "required_columns": required_columns,
                "pit_columns": pit_columns,
                "missing_required_columns": missing,
                "pit_safe": bool(spec.get("pit_safe", False)),
                **{
                    key: value
                    for key, value in spec.items()
                    if key not in {"required_columns", "pit_columns", "pit_safe"}
                },
            }
        )
    return results


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def print_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _load_plan_or_error(identifier: str) -> OrderPlan:
    plans_root = PersistencePaths.from_settings(_settings()).order_plans_root
    try:
        return load_order_plan(
            identifier,
            artifact_store=_artifact_store(plans_root),
        )
    except (StorageError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_plan_events_or_error(
    order_plan_id: str,
    *,
    artifact_store: ArtifactStore,
) -> list[OrderPlanEvent]:
    try:
        return load_order_plan_events(order_plan_id, artifact_store=artifact_store)
    except (StorageError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _parse_latest_limit(value: str) -> int:
    if value.startswith("latest-"):
        return int(value.removeprefix("latest-"))
    return int(value)


def _parse_json_params(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON params: {exc}") from exc
