"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from qmt_agent_trader.agent.workflows.factor_discovery import run_factor_discovery
from qmt_agent_trader.agent.workflows.strategy_discovery import run_strategy_discovery
from qmt_agent_trader.backtest.service import compare_backtest_reports, run_backtest_report
from qmt_agent_trader.broker.order_plan import OrderPlan
from qmt_agent_trader.broker.remote_client import RemoteQMTBrokerClient
from qmt_agent_trader.broker.risk import run_order_plan_risk_checks
from qmt_agent_trader.cli.tui import run_tui
from qmt_agent_trader.core.audit import AuditLogger
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.core.types import ApprovalStatus, RiskStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient
from qmt_agent_trader.factors.service import compute_factor_to_lake, validate_factor
from qmt_agent_trader.services.data_update_service import (
    TushareDataUpdateService,
    build_data_update_plan,
)
from qmt_agent_trader.services.order_plan_service import (
    build_sample_paper_order_plan,
    load_order_plan,
    save_order_plan,
)
from qmt_agent_trader.strategy.approval import StrategyApproval, write_approval_file

app = typer.Typer(help="Mac control plane for QMT agent trading.")
data_app = typer.Typer(help="Data lake commands.")
factor_app = typer.Typer(help="Factor commands.")
backtest_app = typer.Typer(help="Backtest commands.")
agent_app = typer.Typer(help="LLM research agent commands.")
strategy_app = typer.Typer(help="Strategy approval commands.")
broker_app = typer.Typer(help="Remote QMT broker commands.")
trade_app = typer.Typer(help="Order plan and trading commands.")

app.add_typer(data_app, name="data")
app.add_typer(factor_app, name="factor")
app.add_typer(backtest_app, name="backtest")
app.add_typer(agent_app, name="agent")
app.add_typer(strategy_app, name="strategy")
app.add_typer(broker_app, name="broker")
app.add_typer(trade_app, name="trade")


def _settings() -> Settings:
    return get_settings()


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


def _tushare_client() -> TushareClient:
    settings = _settings()
    token = settings.tushare_token.get_secret_value() if settings.tushare_token else None
    return TushareClient(token=token)


def _data_lake() -> DataLake:
    settings = _settings()
    return DataLake(
        root=settings.resolved_data_dir / "lake",
        duckdb_path=settings.resolved_data_dir / "qmt_agent_trader.duckdb",
    )


def _audit_logger(name: str) -> AuditLogger:
    settings = _settings()
    return AuditLogger(settings.resolved_log_dir / "audit" / f"{name}.jsonl")


@app.command()
def tui() -> None:
    """Run the Textual TUI skeleton."""
    run_tui()


@data_app.command("update")
def data_update(
    from_date: Annotated[str, typer.Option("--from")],
    to_date: Annotated[str, typer.Option("--to")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    skip_daily: Annotated[bool, typer.Option("--skip-daily")] = False,
    skip_basics: Annotated[bool, typer.Option("--skip-basics")] = False,
) -> None:
    """Update the local data lake from Tushare Pro."""
    client = _tushare_client()
    if dry_run:
        plan = build_data_update_plan(client, from_date, to_date)
        print_json({"status": "planned", "requests": plan})
        return
    result = TushareDataUpdateService(client, _data_lake()).update(
        from_date,
        to_date,
        include_daily=not skip_daily,
        include_basics=not skip_basics,
    )
    print_json(result.as_dict())


@data_app.command("validate")
def data_validate() -> None:
    """Validate local data lake artifacts."""
    lake = _data_lake()
    expected = [
        lake.dataset_path("raw", "tushare_trade_calendar"),
        lake.dataset_path("raw", "tushare_stock_basic"),
        lake.dataset_path("raw", "tushare_etf_basic"),
    ]
    missing = [str(path) for path in expected if not path.exists()]
    print_json(
        {
            "status": "ok" if not missing else "missing_data",
            "missing": missing,
            "duckdb_exists": lake.duckdb_path.exists(),
        }
    )


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
            reports_dir=Path("reports/backtests"),
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
    print_json(compare_backtest_reports(Path("reports/backtests"), limit=_parse_latest_limit(runs)))


@agent_app.command("discover-factors")
def discover_factors(theme: Annotated[str, typer.Option("--theme")]) -> None:
    print_json(run_factor_discovery(theme, settings=_settings()))


@agent_app.command("discover-strategies")
def discover_strategies(universe: Annotated[str, typer.Option("--universe")]) -> None:
    print_json(run_strategy_discovery(universe, settings=_settings()))


@strategy_app.command("list")
def strategy_list() -> None:
    approval_dir = Path("approvals")
    files = (
        sorted(str(path) for path in approval_dir.glob("*.approval.yaml"))
        if approval_dir.exists()
        else []
    )
    print_json({"approvals": files})


@strategy_app.command("review")
def strategy_review(strategy_id: Annotated[str, typer.Option("--strategy-id")]) -> None:
    print_json({"strategy_id": strategy_id, "status": ApprovalStatus.REVIEW_REQUIRED})


@strategy_app.command("approve")
def strategy_approve(
    strategy_id: Annotated[str, typer.Option("--strategy-id")],
    paper_only: bool = True,
) -> None:
    approval = StrategyApproval(
        strategy_id=strategy_id,
        strategy_name=strategy_id.replace("_", " ").title(),
        strategy_version="1.0.0",
        approved_by="human",
        allowed_universe=["A_SHARE_STOCK", "ETF"],
        allowed_accounts=["default_stock_account"],
        max_single_position_pct=0.10,
        max_turnover_daily_pct=0.30,
        max_order_value=100000,
        live_trading_allowed=False,
        paper_trading_allowed=True,
        notes="First approval for paper trading only.",
    )
    path = write_approval_file(approval, Path("approvals"))
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
    path = save_order_plan(plan, Path("order_plans"))
    print_json({"status": "generated", "path": str(path), "plan_hash": plan.plan_hash})


@trade_app.command("risk-check")
def trade_risk_check(plan: Annotated[str, typer.Option("--plan")]) -> None:
    order_plan = _load_plan_or_error(plan)
    result = run_order_plan_risk_checks(order_plan)
    payload = {
        "plan": plan,
        "order_plan_id": order_plan.order_plan_id,
        "status": result.status,
        "checks": [check.model_dump(mode="json") for check in result.checks],
    }
    _audit_logger("trade").append("trade.risk_check", "cli", payload)
    print_json(payload)


@trade_app.command("paper")
def trade_paper(plan: Annotated[str, typer.Option("--plan")]) -> None:
    order_plan = _load_plan_or_error(plan)
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
    }
    _audit_logger("trade").append("trade.paper", "cli", payload)
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


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _load_plan_or_error(identifier: str) -> OrderPlan:
    try:
        return load_order_plan(identifier)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _parse_latest_limit(value: str) -> int:
    if value.startswith("latest-"):
        return int(value.removeprefix("latest-"))
    return int(value)
