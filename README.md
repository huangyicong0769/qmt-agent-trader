# qmt-agent-trader

`qmt-agent-trader` is an in-progress QMT trading-agent research project. The
intended architecture is a Mac-side control plane plus a Windows-only QMT
gateway, with the Mac side handling research, data preparation, backtesting,
strategy review, risk checks, and order-plan generation.

The repository is not production-ready. Many pieces are scaffolds, experiments,
or partially integrated workflows. Treat the current code as a development
workspace for building and testing the architecture, not as a finished trading
system.

> This project is not investment advice. Do not use it for live trading without
> an independent security, risk, and broker-integration review.

## Current status

The project currently contains working code and tests for the main module
boundaries, but the end-to-end product is still under development.

Implemented or partially implemented surfaces include:

- Mac-side Typer CLI entry point: `qmt-agent`.
- Local data-lake commands backed by DuckDB and Parquet.
- Registry-driven Tushare capability discovery, fetch planning, local fetch execution,
  and data-table builds.
- Factor, backtest, strategy, broker, and order-plan modules.
- LLM agent runtime experiments with tool permissions and audit boundaries.
- NiceGUI web-studio and Textual TUI entry points.
- A Windows gateway package intended to isolate MiniQMT / `xtquant`.
- Unit tests covering the current module contracts.

Important unfinished areas:

- The public onboarding path still needs sample data, screenshots, and a
  release smoke test.
- Live trading is intentionally gated and should be considered unvalidated.
- The Windows gateway requires real MiniQMT / QMT validation on a Windows host.
- Generated strategies and agent outputs still need stronger review and
  promotion workflows.
- CI, release packaging, and contributor documentation are not finished.

## Repository layout

```text
.
├── configs/                         # YAML defaults and local config examples
├── gateway/windows_qmt_gateway/      # Windows-only QMT gateway package
├── scripts/                          # Development and research helper scripts
├── src/qmt_agent_trader/
│   ├── agent/                        # LLM runtime, tools, permissions, workflows
│   ├── backtest/                     # Backtest engine, reports, constraints
│   ├── broker/                       # Order plans, risk, remote gateway client
│   ├── cli/                          # Typer CLI and Textual TUI
│   ├── core/                         # Settings, audit, security, shared types
│   ├── data/                         # Data lake storage, loaders, transforms
│   ├── factors/                      # Factor registry, library, validation
│   ├── services/                     # Data, order-plan, report, scheduler services
│   ├── strategy/                     # Strategy specs, registry, approvals
│   └── web/                          # FastAPI / NiceGUI web studio
└── tests/                            # Unit and API coverage
```

Local runtime directories are intentionally ignored by Git: `data/`, `logs/`,
`reports/`, `sessions/`, `order_plans/`, `approvals/`, and
`src/qmt_agent_trader/agent/generated/`.

## Requirements

- Python 3.11+.
- [`uv`](https://docs.astral.sh/uv/) for Python environment management.
- Tushare Pro token if you want to run remote market-data updates.
- Optional DeepSeek-compatible API credentials for LLM-agent experiments.
- Optional Windows machine with MiniQMT / QMT and vendor-provided `xtquant` for
  gateway validation.

Do not install unknown `xtquant` packages from PyPI. The gateway is intended to
load the `xtquant` package from the local QMT installation.

## Quick start

```bash
git clone <repo-url>
cd qmt-agent-trader

uv sync
cp .env.example .env
uv run qmt-agent --help
```

Only set credentials for the workflows you are actively testing:

```dotenv
TUSHARE_TOKEN=
DEEPSEEK_API_KEY=
QMT_GATEWAY_BASE_URL=http://192.168.1.100:8765
QMT_GATEWAY_API_KEY=
QMT_GATEWAY_HMAC_SECRET=

DRY_RUN=true
LIVE_TRADING_ENABLED=false
```

`.env` and local MCP server configuration are ignored by Git.

## Useful commands

```bash
# Inspect CLI surfaces
uv run qmt-agent --help
uv run qmt-agent agent --help
uv run qmt-agent strategy --help
uv run qmt-agent trade --help

# Local UI entry points
uv run qmt-agent web --host 127.0.0.1 --port 7860
uv run qmt-agent tui

# Data lake
uv run qmt-agent data validate
uv run qmt-agent data capabilities --category market
uv run qmt-agent data plan-fetch --api daily --symbols 000001.SZ --from 20240101 --to 20240131
uv run qmt-agent data migrate-new-layout

# Agent experiments
uv run qmt-agent agent tools
uv run qmt-agent agent ask --prompt "research a momentum reversal idea" --json
uv run qmt-agent agent run-factor-discovery --theme "low volatility quality"

# Backtest and strategy surfaces
uv run qmt-agent backtest run --symbol 000001.SZ --signal-date 20240102
uv run qmt-agent strategy list
uv run qmt-agent strategy candidates

# Order-plan surfaces
uv run qmt-agent trade generate-plan --strategy-id <strategy_id>
uv run qmt-agent trade risk-check --plan <order_plan_path_or_id>
```

Some commands require local data, configured API credentials, or generated
strategy/order-plan artifacts. The README examples are entry points, not a full
demo flow yet.

## Safety posture

The current default configuration is deliberately conservative:

- `DRY_RUN=true`.
- `LIVE_TRADING_ENABLED=false`.
- The Mac side produces order-plan artifacts before any broker action.
- Live submit refuses execution unless live trading is enabled and the caller
  provides explicit confirmation.
- Agent-visible tools are permissioned and audited.
- Registry-driven Tushare fetch tools enforce rate limits, request budgets, schema checks, and
  write locks.

These are engineering guardrails, not proof that the system is safe for real
capital.

## Data workflow

The local data lake lives under `data/` and is not committed. Stable datasets are
written below `data/lake`, with DuckDB metadata in `data/qmt_agent_trader.duckdb`.

```bash
uv run qmt-agent data capabilities --category market
uv run qmt-agent data plan-fetch --api daily --symbols 000001.SZ --from 20240101 --to 20240131
uv run qmt-agent data fetch --api daily --symbols 000001.SZ --from 20240101 --to 20240131 --execute-plan
uv run qmt-agent data build-table --table daily_market
uv run qmt-agent data validate
```

Legacy Tushare raw files such as `raw/tushare_daily.parquet` and
`raw/tushare_daily_20240101_20240131.parquet` are one-way migrated into the new
`raw/tushare/*.parquet` layout with:

```bash
uv run qmt-agent data migrate-new-layout
```

Use `--keep-legacy` only for an explicit audit snapshot. The runtime query and
Agent fetch tools do not read the old layout.

## LLM and MCP configuration

Set the DeepSeek-compatible environment variables in `.env` before testing live
LLM workflows:

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

Optional MCP tools are configured through an ignored local file:

```bash
cp configs/mcp.servers.example.json configs/mcp.servers.json
```

Then set:

```dotenv
MCP_ENABLED=true
MCP_CONFIG_PATH=configs/mcp.servers.json
TAVILY_API_KEY=
```

Keep MCP tools read-only unless a workflow has a clear permission boundary.

## Windows QMT gateway

Run the gateway only on a Windows host that has MiniQMT / QMT installed.

```powershell
cd gateway/windows_qmt_gateway
uv sync
copy .env.example .env
uv run qmt-gateway --help
uv run qmt-gateway qmt-smoke-test
uv run qmt-gateway serve
```

Important gateway environment variables:

```dotenv
QMT_XTQUANT_PATH=
QMT_MINIQMT_PATH=
QMT_ACCOUNT_ID=
QMT_ACCOUNT_TYPE=STOCK
GATEWAY_API_KEY=
GATEWAY_HMAC_SECRET=
DRY_RUN=true
LIVE_TRADING_ENABLED=false
ALLOW_ORDER_ENDPOINT=false
```

Point the Mac control plane at the gateway with:

```dotenv
QMT_GATEWAY_BASE_URL=http://<windows-host>:8765
QMT_GATEWAY_API_KEY=<same-key>
QMT_GATEWAY_HMAC_SECRET=<same-secret>
```

## Development

Use `uv` for Python environment and command execution.

```bash
uv sync
uv run ruff check .
uv run mypy src
uv run pytest
make check
```

`make check` is the repository acceptance gate. It runs linting, strict mypy for
`src`, and the unit test suite.

Useful focused checks:

```bash
uv run pytest tests/unit/agent
uv run pytest tests/unit/web
uv run pytest tests/unit/test_strategy_registry.py
```

## TODO

### Publish-readiness

- [ ] Add a concise architecture diagram for the Mac control plane, local data
  lake, LLM tool runtime, and Windows QMT gateway boundary.
- [ ] Add a sample non-secret `.env` walkthrough for common modes:
  data-only, LLM research, paper trading, and Windows gateway integration.
- [ ] Add a release smoke-test script that verifies CLI help, data validation,
  strategy listing, and paper order-plan generation on a clean checkout.
- [ ] Document how to rotate gateway API keys and HMAC secrets.
- [ ] Add screenshots or a short walkthrough for the NiceGUI web studio once the
  UI is ready for external readers.

### Safety and operations

- [ ] Add an operations runbook for starting, stopping, and health-checking the
  Windows gateway.
- [ ] Add a failure-mode guide for stale market data, partial data coverage,
  Tushare fetch timeouts, and gateway connectivity errors.
- [ ] Add an audit-log retention policy for `logs/audit/*.jsonl`.
- [ ] Define a formal promotion path from generated strategies to reviewed,
  tracked strategy modules.
- [ ] Add an example paper-trading approval file with clearly fake account and
  strategy identifiers.

### Research and strategy roadmap

- [ ] Expand built-in strategy examples beyond the current ETF trend and factor
  rank examples.
- [ ] Add richer benchmark and universe configuration examples.
- [ ] Add report templates for factor discovery, strategy engineering, and
  backtest comparison.
- [ ] Add regression fixtures for representative multi-symbol agent sessions.
- [ ] Add public sample data or synthetic fixtures so contributors can run a
  meaningful demo without private Tushare/QMT data.

### Documentation backlog

- [ ] Add API documentation for agent tool contracts and permission levels.
- [ ] Document local MCP server configuration with read-only defaults and
  security expectations.
- [ ] Document directory ownership for `data/`, `reports/`, `sessions/`,
  `order_plans/`, and `approvals/`.
- [ ] Add a contributor guide covering branch naming, atomic commits, `uv`, and
  the `make check` acceptance gate.
