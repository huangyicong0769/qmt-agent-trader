# qmt-agent-trader

Mac 主控 + Windows QMT Gateway 的程序化交易 Agent 系统。

第一版目标是先交付可运行、可测试、权限边界正确的工程骨架：

- Mac 端负责数据、研究、回测、Agent、审批和订单计划。
- Windows Gateway 是唯一允许加载 `xtquant` / MiniQMT 的进程。
- 默认 `DRY_RUN=true`、`LIVE_TRADING_ENABLED=false`、`ALLOW_ORDER_ENDPOINT=false`。
- LLM Agent 不能直接调用实盘下单接口，订单必须先形成不可变 `OrderPlan` 并通过审批和风控。
- LLM Agent 不能直接调用远程行情客户端，只能通过受控数据工具规划或补齐数据；并发、限速、日期跨度和写入锁由本地代码强制执行。

## Quick Start

```bash
uv run qmt-agent --help
uv run pytest
uv run ruff check .
uv run mypy src
```

Windows Gateway 子项目：

```powershell
cd gateway/windows_qmt_gateway
uv run qmt-gateway --help
uv run qmt-gateway qmt-smoke-test
uv run qmt-gateway serve
```

## Architecture

```text
Mac qmt-agent-trader
  CLI/TUI
  Agent Orchestrator
  Data Lake: DuckDB + Parquet + fetch state
  Backtest + Leakage Checks
  Strategy Approval
  OrderPlan + Risk
  RemoteQMTBrokerClient
        |
        | LAN HTTP/WebSocket, API key + HMAC + timestamp + nonce
        v
Windows QMT Gateway
  FastAPI
  xtquant loader
  QMT adapter
  Gateway risk
  SQLite/JSONL audit
```

## Safety Defaults

The repository never stores real account IDs, tokens, or secrets. Copy `.env.example`
to `.env` locally and keep `.env` untracked.

## Controlled Data Updates

Remote data updates use the local `TushareDataUpdateService` and write stable
incremental datasets such as `raw/tushare_daily.parquet`. The Agent-facing
tools are `plan_remote_data_update` and `run_remote_data_update`; dry-run calls
only return a plan, while live calls require `TUSHARE_TOKEN` and use the local
rate limiter, update lock, and DuckDB fetch-state tables.
