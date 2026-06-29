from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.schemas import ExperimentStatus, ToolContext
from qmt_agent_trader.agent.workflows.strategy_engineering import StrategyEngineeringWorkflow


class FakeStrategyRegistry:
    def __init__(self, code_path: Path) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.code_path = code_path
        self.code_path.write_text(
            "def generate_signals(context):\n    return context.bars\n",
            encoding="utf-8",
        )

    def run_tool(
        self,
        name: str,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        self.calls.append((name, input_data))
        if name == "create_strategy_spec":
            return {
                "status": "created",
                "strategy_spec": {
                    "strategy_id": "strat_replay",
                    "name": "Replay",
                    "factors": [{"factor_id": "momentum_20d", "weight": 1.0}],
                },
            }
        if name == "generate_strategy_code":
            return {
                "status": "generated",
                "code_path": str(self.code_path),
                "tests_path": str(self.code_path.with_name("test_strategy.py")),
            }
        if name == "run_strategy_static_checks":
            return {"status": "PASSED", "issues": []}
        if name == "run_backtest":
            return {
                "status": "completed",
                "report_path": "reports/research/replay.json",
                "metrics": {"sharpe": 1.0},
                "diagnostics": {"status": "PASS"},
            }
        if name == "generate_research_report":
            return {"report_path": "reports/research/replay.md"}
        raise AssertionError(name)


def test_strategy_workflow_passes_spec_code_path_and_factor_to_backtest(tmp_path) -> None:
    fake = FakeStrategyRegistry(tmp_path / "strategy.py")
    store = ExperimentStore(tmp_path / "experiments")
    workflow = StrategyEngineeringWorkflow(fake, store)  # type: ignore[arg-type]

    exp = workflow.run("idea", ["momentum_20d"], "stock_etf", "20200101", "20240101")

    backtest_call = next(payload for name, payload in fake.calls if name == "run_backtest")
    assert exp.status == ExperimentStatus.REVIEW_REQUIRED
    assert backtest_call["strategy_spec"]["strategy_id"] == "strat_replay"
    assert backtest_call["code_path"] == str(fake.code_path)
    assert backtest_call["factor_name"] == "momentum_20d"


def test_strategy_workflow_stops_when_static_check_fails(tmp_path) -> None:
    fake = FakeStrategyRegistry(tmp_path / "strategy.py")

    def failing_run_tool(
        name: str,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        if name == "run_strategy_static_checks":
            fake.calls.append((name, input_data))
            return {"status": "FAILED", "issues": ["danger"]}
        return FakeStrategyRegistry.run_tool(fake, name, input_data, context)

    fake.run_tool = failing_run_tool  # type: ignore[method-assign]
    workflow = StrategyEngineeringWorkflow(fake, ExperimentStore(tmp_path / "experiments"))  # type: ignore[arg-type]

    exp = workflow.run("idea", ["momentum_20d"], "stock_etf", "20200101", "20240101")

    assert exp.status == ExperimentStatus.FAILED
    assert not any(name == "run_backtest" for name, _ in fake.calls)
